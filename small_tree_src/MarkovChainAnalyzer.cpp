#include <exception>
#include <Eigen/SparseLU>

#if defined(__APPLE__) && !defined(NO_ACCELERATE)
    #define MCA_USE_ACCELERATE 1
    // Opt into Accelerate's new, self-consistent BLAS/LAPACK declarations. On recent macOS SDKs the
    // umbrella <Accelerate/Accelerate.h> otherwise pulls in the legacy Fortran BLAS header alongside
    // the new one and they collide ("Conflicting types for 'saxpy_'"). We use only the Sparse BLAS
    // (SparseMultiply etc.), which is unaffected by this switch, so it is safe to set. Build with
    // -DNO_ACCELERATE to skip Accelerate entirely and use the (numerically identical) CPU SpMV.
    #ifndef ACCELERATE_NEW_LAPACK
        #define ACCELERATE_NEW_LAPACK 1
    #endif
    #include <Accelerate/Accelerate.h>
#else
    #define MCA_USE_ACCELERATE 0
#endif

// Dense LAPACK. The full eigensystem of the symmetrized kernel is an O(n^3) calculation that Eigen's
// SelfAdjointEigenSolver performs on a single core: 661 s at n = 10395, the 8-taxon state space.
// Accelerate's divide-and-conquer dsyevd is multithreaded and about ten times faster. The legacy
// Fortran symbols are declared here directly, so the Accelerate umbrella header (whose BLAS
// declarations collided on a beta SDK, hence NO_ACCELERATE) is not needed. Build with -DNO_LAPACK to
// use Eigen instead, as on a machine without Accelerate.
#if defined(__APPLE__) && !defined(NO_LAPACK)
    #define MCA_USE_LAPACK 1
    #if !MCA_USE_ACCELERATE
    extern "C" {
        void dsyevd_(const char* jobz, const char* uplo, const int* n, double* a, const int* lda, double* w,
                     double* work, const int* lwork, int* iwork, const int* liwork, int* info);
        void dgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k,
                    const double* alpha, const double* a, const int* lda, const double* b, const int* ldb,
                    const double* beta, double* c, const int* ldc);
    }
    #endif
#else
    #define MCA_USE_LAPACK 0
#endif
#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <limits>
#include <numeric>
#include <random>
#include <chrono>

#include <set>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include "MarkovChainAnalyzer.hpp"
#include "Threads.hpp"
#include "TreeCache.hpp"

namespace {

    double quietNaNValue(void) {

        return std::numeric_limits<double>::quiet_NaN();
    }

    [[maybe_unused]] bool finitePositive(double x) {

        return std::isfinite(x) && x > 0.0;
    }

}



MarkovChainAnalyzer::MarkovChainAnalyzer(ThreadPool* tp, TreeCache* cache, std::string nme, bool useSparse)
    : threadPool(tp), name(nme), isSparse(useSparse) {
    
    if (!cache)
        throw std::invalid_argument("Null TreeCache");
    buildFromCache(cache);
    initCommon();
}

MarkovChainAnalyzer::MarkovChainAnalyzer(ThreadPool* tp, const SparseMatrix& P, const Vector& pi)
    : threadPool(tp), P_row_sparse(P), pi(pi), n(P.rows()), isSparse(true) {
    
    P_row_sparse.makeCompressed();
    stateHashes.resize(static_cast<size_t>(n), 0);
    initCommon();
}

MarkovChainAnalyzer::MarkovChainAnalyzer(ThreadPool* tp, const DenseMatrix& P, const Vector& pi)
    : threadPool(tp), P_dense(P), pi(pi), n(P.rows()), isSparse(false) {
    
    stateHashes.resize(static_cast<size_t>(n), 0);
    initCommon();
}

// Apply (I - S) y, where S = D^{1/2} P D^{-1/2} and D = diag(pi). Reversibility, pi_i P_ij =
// pi_j P_ji, makes S symmetric, and this operator is therefore symmetric and positive
// semi-definite: its eigenvalues are 1 - lambda_i, and its only null direction is sqrt(pi). The
// mean first-passage times reduce to a linear system in this operator, as does the asymptotic
// variance of a functional, so a single conjugate-gradient solver serves both. The operator is
// never formed; each application costs one product with the sparse transition kernel.
namespace {

    // Every iteration of a Lanczos eigensolve, and every iteration of a conjugate-gradient solve,
    // costs one product of the kernel with a vector. For a TBR analysis of ten taxa the kernel has
    // some 590 million entries and a single such product takes seconds, so a solve that needs a few
    // hundred of them takes a quarter of an hour. Eigen does not parallelize a sparse product with a
    // vector -- it threads dense matrix multiplication, but not this -- and the product is where
    // essentially all of the time goes.
    //
    // The rows of a row-major sparse matrix are independent: the ith entry of the result is formed
    // from the ith row alone, and nothing is written twice. The product therefore parallelizes with
    // no locking and no communication, and is limited only by memory bandwidth.
    class SparseMatVecTask : public ThreadTask {

        public:
                                    SparseMatVecTask(void) = delete;
                                    SparseMatVecTask(const int* out, const int* in, const double* v,
                                                     const double* x, double* y,
                                                     Eigen::Index lo, Eigen::Index hi, const double* scale = nullptr) :
                                        outerIndex(out), innerIndex(in), values(v),
                                        xIn(x), yOut(y), begin(lo), end(hi), rowScale(scale) { }
            void                    run(void) override;

        private:
            const int*              outerIndex;
            const int*              innerIndex;
            const double*           values;
            const double*           xIn;
            double*                 yOut;
            Eigen::Index            begin;
            Eigen::Index            end;
            const double*           rowScale;                     // optional: y_i is scaled by rowScale[i]
    };

    void SparseMatVecTask::run(void) {

        for (Eigen::Index i = begin; i < end; ++i)
            {
            double sum = 0.0;
            for (int k = outerIndex[i]; k < outerIndex[i+1]; ++k)
                sum += values[k] * xIn[innerIndex[k]];
            yOut[i] = (rowScale != nullptr) ? rowScale[i] * sum : sum;
            }
    }

    // A matrix operation for Spectra that forms the product in parallel. Spectra asks only that an
    // operation report its dimensions and multiply a vector; it does not care how the product is
    // computed. Its own SparseSymMatProd defers to Eigen, whose sparse product with a vector runs on
    // a single core, and that product is where essentially all of the time of a Lanczos eigensolve is
    // spent: for a TBR analysis of ten taxa one product takes seconds, and a solve needs hundreds.
    //
    // The matrix supplied here is the symmetrized kernel, S = D^{1/2} P D^{-1/2}, which is symmetric
    // by construction. For a symmetric matrix the compressed-column arrays are the compressed-row
    // arrays, so the column-major matrix Eigen has already built can be traversed by rows -- and
    // therefore multiplied in parallel, one row per unit of work -- without being copied or
    // transposed.
    // Spectra operator that defers the product to the analyzer's applyS, so the eigensolves share the
    // one accelerated (or threaded) symmetric multiply used by every other spectral routine.
    //
    // The operator also carries an optional deadline. Spectra's restart loop cannot be paused, and
    // re-entering it in batches rebuilds the Krylov basis each time, which both costs products and can
    // steer the solve onto an interior eigenvalue. Instead each product checks the clock, and once the
    // deadline has passed it throws SolveBudgetSpent, which unwinds the solve from inside.
    struct SolveBudgetSpent { };

    class AnalyzerSymOp {

        public:
            using Scalar = double;
            using Clock  = std::chrono::steady_clock;
                                    AnalyzerSymOp(void) = delete;
                                    AnalyzerSymOp(const MarkovChainAnalyzer* a, Eigen::Index dim) : self(a), n(dim) { }
            Eigen::Index            rows(void) const { return n; }
            Eigen::Index            cols(void) const { return n; }
            void                    perform_op(const double* x_in, double* y_out) const;
            void                    setDeadline(const Clock::time_point* d) const { deadline = d; }

        private:
            const MarkovChainAnalyzer* self;
            Eigen::Index               n;
            mutable const Clock::time_point* deadline = nullptr;
    };

    class DeflatedSymOp {

        public:
            using Scalar = double;
                                    DeflatedSymOp(void) = delete;
                                    DeflatedSymOp(const AnalyzerSymOp& base, const Eigen::VectorXd& wUnit) :
                                        op(&base), w(&wUnit) { }
            Eigen::Index            rows(void) const { return op->rows(); }
            Eigen::Index            cols(void) const { return op->cols(); }
            void                    perform_op(const double* x_in, double* y_out) const
                                        {
                                        Eigen::Index N = op->rows();
                                        op->perform_op(x_in, y_out);                 // y = S x
                                        double dot = 0.0;                            // dot = w . x
                                        const double* wp = w->data();
                                        for (Eigen::Index i = 0; i < N; ++i)
                                            dot += wp[i] * x_in[i];
                                        for (Eigen::Index i = 0; i < N; ++i)         // y = S x - w (w.x)
                                            y_out[i] -= wp[i] * dot;
                                        }

        private:
            const AnalyzerSymOp* op;
            const Eigen::VectorXd*    w;
    };

    // S + sigma I, for the solve of the most negative eigenvalue. Spectra stops when a Ritz pair's residual
    // falls below tol * |theta|, a tolerance relative to the eigenvalue itself. Where lambda_min sits in
    // the dense cluster just below zero -- -1e-4 on a 9-taxon TBR kernel -- that asks for a residual near
    // 1e-11, which the cluster cannot deliver in any reasonable number of restarts: a 10-taxon TBR solve
    // ran 300 restarts (400 s) without converging. Shifted by sigma = 2 the spectrum lies in [1, 3] and
    // the same tolerance is, in effect, an absolute one of 1e-7 to 3e-7 on lambda_min wherever it lies.
    // (A shift of 1 would put a lambda_min near -1, a chain flipping between two states, back at zero.)
    // The Ritz vectors are unchanged and the Ritz values move by exactly sigma.
    class ShiftedSymOp {

        public:
            using Scalar = double;
                                    ShiftedSymOp(void) = delete;
                                    ShiftedSymOp(const AnalyzerSymOp& base, double s) : op(&base), sigma(s) { }
            Eigen::Index            rows(void) const { return op->rows(); }
            Eigen::Index            cols(void) const { return op->cols(); }
            void                    perform_op(const double* x_in, double* y_out) const
                                        {
                                        op->perform_op(x_in, y_out);                 // y = S x
                                        Eigen::Index N = op->rows();
                                        for (Eigen::Index i = 0; i < N; ++i)         // y = (S + sigma I) x
                                            y_out[i] += sigma * x_in[i];
                                        }

        private:
            const AnalyzerSymOp* op;
            double               sigma;
    };

    // The inverse of A = I - S + w w', applied by a conjugate-gradient solve, for a shift-invert Lanczos
    // solve of the slow end of the spectrum. A maps the eigenvalue lambda of S to 1 - lambda (and w to
    // 1), so A^{-1} maps the slow modes, lambda near 1, to its largest and best-separated eigenvalues,
    // mu = 1 / (1 - lambda). The cluster just below 1 that stalls an ordinary Lanczos solve of S is
    // spread out by the inversion, and a small subspace resolves it in a few dozen solves.
    class ShiftInvertOp {

        public:
            using Scalar = double;
                                    ShiftInvertOp(void) = delete;
                                    ShiftInvertOp(const std::function<void(const double*, double*)>* f, Eigen::Index dim) :
                                        solve(f), n(dim) { }
            Eigen::Index            rows(void) const { return n; }
            Eigen::Index            cols(void) const { return n; }
            void                    perform_op(const double* x_in, double* y_out) const { (*solve)(x_in, y_out); }

        private:
            const std::function<void(const double*, double*)>* solve;
            Eigen::Index            n;
    };

    // The product of the transition kernel with a block of vectors, Y = P X, X and Y stored row-major
    // (n x numCols) so that each kernel entry is read once and applied to every column. The product is
    // limited by memory bandwidth, so a block of k vectors costs little more than a single vector.
    class SparseMatBlockTask : public ThreadTask {

        public:
                                    SparseMatBlockTask(void) = delete;
                                    SparseMatBlockTask(const int* out, const int* in, const double* v,
                                                       const double* x, double* y, int k,
                                                       Eigen::Index lo, Eigen::Index hi, const double* scale = nullptr) :
                                        outerIndex(out), innerIndex(in), values(v),
                                        xIn(x), yOut(y), numCols(k), begin(lo), end(hi), rowScale(scale) { }
            void                    run(void) override;

        private:
            const int*              outerIndex;
            const int*              innerIndex;
            const double*           values;
            const double*           xIn;
            double*                 yOut;
            int                     numCols;
            Eigen::Index            begin;
            Eigen::Index            end;
            const double*           rowScale;                     // optional: Y row i is scaled by rowScale[i]
    };

    void SparseMatBlockTask::run(void) {

        for (Eigen::Index i = begin; i < end; ++i)
            {
            double* yi = yOut + i * numCols;
            for (int c = 0; c < numCols; ++c)
                yi[c] = 0.0;
            for (int k = outerIndex[i]; k < outerIndex[i+1]; ++k)
                {
                double v = values[k];
                const double* xj = xIn + static_cast<Eigen::Index>(innerIndex[k]) * numCols;
                for (int c = 0; c < numCols; ++c)
                    yi[c] += v * xj[c];
                }
            if (rowScale != nullptr)
                {
                double si = rowScale[i];
                for (int c = 0; c < numCols; ++c)
                    yi[c] = si * yi[c];
                }
            }
    }

    // body(lo, hi, part) over a fixed partition of the rows [0, N) into `parts` ranges, on the thread
    // pool when there is one. Reductions are accumulated per part and combined in part order by the
    // caller, so a result does not depend on the number of threads or on how the parts are scheduled.
    class RowRangeTask : public ThreadTask {

        public:
                                    RowRangeTask(const std::function<void(Eigen::Index, Eigen::Index, int)>* f,
                                                 Eigen::Index lo, Eigen::Index hi, int p) :
                                        body(f), begin(lo), end(hi), part(p) { }
            void                    run(void) override
                                        {
                                        try
                                            {
                                            (*body)(begin, end, part);
                                            }
                                        catch (...)
                                            {
                                            failure = std::current_exception();
                                            }
                                        }
            std::exception_ptr      failure;                      // rethrown on the calling thread

        private:
            const std::function<void(Eigen::Index, Eigen::Index, int)>* body;
            Eigen::Index            begin;
            Eigen::Index            end;
            int                     part;
    };

    void forRowRanges(ThreadPool* pool, Eigen::Index N, int parts,
                      const std::function<void(Eigen::Index, Eigen::Index, int)>& body) {

        Eigen::Index chunk = (N + parts - 1) / parts;
        std::vector<RowRangeTask> tasks;
        tasks.reserve(static_cast<size_t>(parts));
        for (int p = 0; p < parts; ++p)
            {
            Eigen::Index lo = std::min(N, static_cast<Eigen::Index>(p) * chunk);
            Eigen::Index hi = std::min(N, lo + chunk);
            tasks.emplace_back(&body, lo, hi, p);
            }
        if (pool == nullptr)
            {
            for (RowRangeTask& t : tasks)
                t.run();
            }
        else
            {
            for (RowRangeTask& t : tasks)
                pool->pushTask(&t);
            pool->wait();
            }
        // The pool's workers report and swallow an exception; a part that failed would leave its rows
        // unwritten (the kernel build fills preallocated storage), so the failure is raised here instead.
        for (RowRangeTask& t : tasks)
            if (t.failure)
                std::rethrow_exception(t.failure);
    }

} // anonymous namespace

double MarkovChainAnalyzer::s_solveBudgetSeconds = 1800.0;

void AnalyzerSymOp::perform_op(const double* x_in, double* y_out) const {

    if (deadline != nullptr && Clock::now() >= *deadline)
        throw SolveBudgetSpent();
    self->applyS(x_in, y_out);
}

void MarkovChainAnalyzer::applySymmetrizedLaplacian(const Vector& piSqrt, const Vector& y, Vector& out) const {

    (void)piSqrt;                                                 // S is applied directly via applyS now
    Eigen::Index N = y.size();
    Vector Sy(N);
    applyS(y.data(), Sy.data());
    out = y - Sy;                                                 // (I - S) y
}

void MarkovChainAnalyzer::applyS(const double* x, double* y) const {

    Eigen::Index N = static_cast<Eigen::Index>(n);

#if MCA_USE_ACCELERATE
    if (isSparse)
        {
        if (accelState == 0)
            buildAcceleratedOperator();
        if (accelState == 1)
            {
            SparseMatrix_Double* A = reinterpret_cast<SparseMatrix_Double*>(accelHandle);
            DenseVector_Double X { static_cast<int>(N), const_cast<double*>(x) };
            DenseVector_Double Y { static_cast<int>(N), y };
            SparseMultiply(*A, X, Y);                             // symmetric upper-triangle product (fp64, threaded)
            return;
            }
        }
#endif

    // CPU path (also the self-check reference): S x = D^{1/2} P D^{-1/2} x.
    if (!piSqrtReady)
        {
        piSqrtCached = posteriorSqrt();
        piSqrtReady = true;
        }
    if (isSparse && threadPool != nullptr)
        {
        // The scalings are done in the threaded passes, into a buffer kept between calls: D^{-1/2} x in
        // a pass of its own and D^{1/2} on each entry of the product as it is formed. Done by Eigen, as
        // earlier, they were two serial passes and two fresh n-vectors per product, at ten taxa a
        // sizeable share of a product with the NNI kernel. The arithmetic is the same.
        ensureRowSparse();
        applyScratch.resize(N);
        const double* sp = piSqrtCached.data();
        double*       tp = applyScratch.data();
        forRowRanges(N >= (Eigen::Index(1) << 16) ? threadPool : nullptr, N, 64, [&](Eigen::Index lo, Eigen::Index hi, int)
            {
            for (Eigen::Index i = lo; i < hi; ++i)
                tp[i] = x[i] / sp[i];
            });

        const int*    outerIndex = P_row_sparse.outerIndexPtr();
        const int*    innerIndex = P_row_sparse.innerIndexPtr();
        const double* values     = P_row_sparse.valuePtr();
        int numTasks = std::max(1, threadPool->getThreadCount() * 4);  // oversubscribe: rows differ in length
        Eigen::Index chunk = std::max<Eigen::Index>(1, (N + numTasks - 1) / numTasks);
        std::vector<SparseMatVecTask> tasks;
        tasks.reserve(static_cast<size_t>(numTasks) + 1);
        for (Eigen::Index lo = 0; lo < N; lo += chunk)
            tasks.emplace_back(outerIndex, innerIndex, values, tp, y, lo, std::min(N, lo + chunk), sp);
        for (SparseMatVecTask& t : tasks)
            threadPool->pushTask(&t);
        threadPool->wait();
        return;
        }

    Eigen::Map<const Vector> xm(x, N);
    Vector t = xm.array() / piSqrtCached.array();
    Vector u(N);
    if (isSparse)
        sparseMatVec(t, u);
    else
        u.noalias() = P_dense * t;
    Eigen::Map<Vector>(y, N) = piSqrtCached.cwiseProduct(u);
}

// Y = S X for a block of numCols vectors, X and Y row-major (n x numCols). Each pass over the kernel
// serves every column, so a block of probes in the stochastic Kemeny estimate costs about as much
// memory traffic as one probe.
void MarkovChainAnalyzer::applySBlock(const double* X, double* Y, int numCols) const {

    Eigen::Index N = static_cast<Eigen::Index>(n);
    if (!piSqrtReady)
        {
        piSqrtCached = posteriorSqrt();
        piSqrtReady = true;
        }

    using RowBlock = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowBlock> Xm(X, N, numCols);
    Eigen::Map<RowBlock>       Ym(Y, N, numCols);

    if (!isSparse)
        {
        RowBlock T = piSqrtCached.cwiseInverse().asDiagonal() * Xm;   // D^{-1/2} X
        Ym.noalias() = piSqrtCached.asDiagonal() * (P_dense * T);
        return;
        }

    // The two diagonal scalings are formed in the threaded passes rather than by Eigen on one core:
    // D^{-1/2} X in a pass of its own, and D^{1/2} on each row of the product as it is finished.
    RowBlock T(N, numCols);                                       // D^{-1/2} X
    {
        const double* sp = piSqrtCached.data();
        double*       tp = T.data();
        forRowRanges(threadPool, N, 64, [&](Eigen::Index lo, Eigen::Index hi, int)
            {
            for (Eigen::Index i = lo; i < hi; ++i)
                {
                double inv = 1.0 / sp[i];
                for (int c = 0; c < numCols; ++c)
                    tp[i * numCols + c] = inv * X[i * numCols + c];
                }
            });
    }

    ensureRowSparse();
    const int*    outerIndex = P_row_sparse.outerIndexPtr();
    const int*    innerIndex = P_row_sparse.innerIndexPtr();
    const double* values     = P_row_sparse.valuePtr();

    int numTasks = (threadPool == nullptr) ? 1 : std::max(1, threadPool->getThreadCount() * 4);
    Eigen::Index chunk = std::max<Eigen::Index>(1, (N + numTasks - 1) / numTasks);
    std::vector<SparseMatBlockTask> tasks;
    tasks.reserve(static_cast<size_t>(numTasks) + 1);
    for (Eigen::Index lo = 0; lo < N; lo += chunk)
        tasks.emplace_back(outerIndex, innerIndex, values, T.data(), Y, numCols, lo, std::min(N, lo + chunk),
                           piSqrtCached.data());
    if (threadPool == nullptr)
        {
        for (SparseMatBlockTask& t : tasks)
            t.run();
        }
    else
        {
        for (SparseMatBlockTask& t : tasks)
            threadPool->pushTask(&t);
        threadPool->wait();
        }
}

void MarkovChainAnalyzer::buildAcceleratedOperator(void) const {

#if MCA_USE_ACCELERATE
    accelState = -1;
    if (!isSparse || n < 2)
        return;

    Eigen::Index N = static_cast<Eigen::Index>(n);
    if (!piSqrtReady) { piSqrtCached = posteriorSqrt(); piSqrtReady = true; }

    // Upper triangle of S from P (no transient dense S): for i <= j, S_ij = sqrt(pi_i/pi_j) P_ij.
    // Detailed balance makes S_ji equal, so storing one triangle and declaring the matrix symmetric
    // reproduces the full product at half the memory and half the bandwidth -- the win on this
    // bandwidth-bound fp64 kernel. Underflowed states are already gone (support restriction), so every
    // sqrt is finite here.
    long cnt = 0;
    for (int col = 0; col < P_row_sparse.outerSize(); ++col)
        for (RowSparseMatrix::InnerIterator it(P_row_sparse, col); it; ++it)
            if (it.row() <= it.col())
                ++cnt;

    std::vector<int>    rows; rows.reserve(static_cast<size_t>(cnt));
    std::vector<int>    cols; cols.reserve(static_cast<size_t>(cnt));
    std::vector<double> vals; vals.reserve(static_cast<size_t>(cnt));
    for (int col = 0; col < P_row_sparse.outerSize(); ++col)
        for (RowSparseMatrix::InnerIterator it(P_row_sparse, col); it; ++it)
            {
            Eigen::Index i = it.row(), j = it.col();
            if (i > j) continue;
            double pj = pi(j);
            if (!(pj > 0.0)) continue;
            vals.push_back(std::sqrt(pi(i) / pj) * it.value());
            rows.push_back(static_cast<int>(i));
            cols.push_back(static_cast<int>(j));
            }

    SparseAttributes_t attr = SparseAttributes_t();
    attr.kind = SparseSymmetric;
    attr.triangle = SparseUpperTriangle;
    SparseMatrix_Double* A = new SparseMatrix_Double();
    *A = SparseConvertFromCoordinate(static_cast<int>(N), static_cast<int>(N),
                                     cnt, 1, attr, rows.data(), cols.data(), vals.data());
    std::vector<int>().swap(rows);
    std::vector<int>().swap(cols);
    std::vector<double>().swap(vals);
    accelHandle = A;

    // Self-check the Accelerate product against the CPU reference on a random vector.
    std::mt19937_64 gen(0xA11CE5EEDULL);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    Vector xr(N);
    for (Eigen::Index i = 0; i < N; ++i) xr(i) = uni(gen);

    Vector yAcc(N);
    { DenseVector_Double X { static_cast<int>(N), xr.data() };
      DenseVector_Double Y { static_cast<int>(N), yAcc.data() };
      SparseMultiply(*A, X, Y); }

    Vector t = xr.array() / piSqrtCached.array();
    Vector u(N);
    sparseMatVec(t, u);
    Vector yCpu = piSqrtCached.cwiseProduct(u);

    double den = std::max(yCpu.norm(), std::numeric_limits<double>::min());
    spmvSelfCheckError = (yAcc - yCpu).norm() / den;

    if (spmvSelfCheckError < 1e-10)
        {
        accelState = 1;
        std::cerr << "MarkovChainAnalyzer (" << name << "): Accelerate sparse SpMV enabled (self-check rel.err "
                  << spmvSelfCheckError << ", symmetric upper-triangle storage, " << cnt << " stored nonzeros).\n";
        }
    else
        {
        std::cerr << "MarkovChainAnalyzer (" << name << "): Accelerate SpMV self-check FAILED (rel.err "
                  << spmvSelfCheckError << "); using threaded CPU SpMV.\n";
        SparseCleanup(*A);
        delete A;
        accelHandle = nullptr;
        accelState = -1;
        }
#endif
}

MarkovChainAnalyzer::~MarkovChainAnalyzer(void) {

#if MCA_USE_ACCELERATE
    if (accelHandle != nullptr)
        {
        SparseMatrix_Double* A = reinterpret_cast<SparseMatrix_Double*>(accelHandle);
        SparseCleanup(*A);
        delete A;
        accelHandle = nullptr;
        }
#endif
}

double MarkovChainAnalyzer::approximateKemenyConstant(int nev) const {

    // The few-eigenvalue sum is a drastic underestimate of K = sum_{j>=2} 1/(1-lambda_j):
    // the ~n-1 bulk eigenvalues near 0 each contribute ~1. Use stochastic Lanczos quadrature,
    // which estimates the full trace of (I - S)^+ without forming the whole spectrum.
    (void)nev;
    if (n <= denseStateLimit)
        return kemenyConstant();
    return kemenyConstantStochastic();
}

double MarkovChainAnalyzer::asymptoticVariance(const Vector& f) const {

    double tau = integratedAutocorrelationTime(f);
    double var = stationaryVariance(f);
    if (std::isnan(tau) || std::isnan(var))
        return std::numeric_limits<double>::quiet_NaN();
    return var * tau;
}

double MarkovChainAnalyzer::averageAcceptanceRate(void) const {

    double acc = 0.0;
    Eigen::Index N = static_cast<Eigen::Index>(n);
    if (isSparse) 
        {
        for (Eigen::Index i = 0; i < N; ++i)
            acc += pi(i) * (1.0 - P_row_sparse.coeff(i, i));
        } 
    else 
        {
        for (Eigen::Index i = 0; i < N; ++i)
            acc += pi(i) * (1.0 - P_dense(i, i));
        }
    return acc;
}

MarkovChainAnalyzer::Vector MarkovChainAnalyzer::leaveProbabilities(void) const {

    // The probability of leaving each state in one step, 1 - P_ii. For a Metropolis kernel this is
    // the state's overall acceptance probability; it measures how sticky a tree is, and is the local
    // counterpart of the mixing diagnostics reported for the chain as a whole.
    Eigen::Index N = static_cast<Eigen::Index>(n);
    Vector leave(N);
    if (isSparse)
        {
        Vector diag = P_row_sparse.diagonal();
        for (Eigen::Index i = 0; i < N; ++i)
            leave(i) = 1.0 - diag(i);
        }
    else
        {
        for (Eigen::Index i = 0; i < N; ++i)
            leave(i) = 1.0 - P_dense(i, i);
        }
    return leave;
}

void MarkovChainAnalyzer::buildFromCache(TreeCache* cache) {

    TreeCacheMap& tcache = cache->getCache();
    n = tcache.size();
    if (n == 0)
        throw std::runtime_error("Empty TreeCache");

    // The map from tree hash to state index is retained. Trees are identified across the program by
    // hash -- the MapTree class reports the MAP tree and the trees carrying each of its splits that
    // way -- whereas the state index here is an arbitrary artifact of the order in which the cache
    // happened to be traversed. Rebuilding the map on demand, or scanning stateHashes, would make
    // the conversion of a set of trees into an indicator vector quadratic in the number of trees.
    //
    // The index is also written into the TreeInfo itself. The kernel has one edge for every neighbor
    // of every tree -- some 590 million of them for a TBR analysis of ten taxa -- and looking each
    // one up by hash would mean that many random probes into a table of two million entries, nearly
    // all of them cache misses. Reading the index straight from the neighbor costs nothing.
    hashToIndex.clear();
    hashToIndex.reserve(n * 2);
    stateHashes.clear();
    stateHashes.resize(n, 0);
    Eigen::Index idx = 0;
    for (const auto& [h, info] : tcache) 
        {
        if (info)
            {
            hashToIndex[h] = idx;
            stateHashes[static_cast<size_t>(idx)] = h;
            info->stateIndex = static_cast<int64_t>(idx);
            ++idx;
            }
        }

    // ---- Restrict to the connected support of the posterior ----
    // A topology whose posterior has underflowed to (near) zero becomes an isolated absorbing
    // self-loop in the kernel: no mass flows into it (its acceptance ratio is zero), and the
    // symmetrization's 1/sqrt(pi) is unbounded. Left in, these make the kernel reducible with an
    // eigenvalue-1 multiplicity in the hundreds of thousands, which defeats the Lanczos solver
    // ("TridiagEigen: eigen decomposition failed"). They carry no posterior mass, so we drop every
    // state with pi <= tau and keep the connected component of the MAP tree, which is irreducible by
    // construction. The inter-basin saddles of a real posterior sit far above tau, so this does not
    // sever basins; any live mass that IS stranded in a separate component is reported, because that
    // is a genuine near-reducibility of the chain rather than a numerical artifact.
    const double supportTau = 1.0e-300;
    size_t numUnderflow = 0;
    if (isSparse)
        {
        for (const auto& [h, info] : tcache)
            if (info && !(info->posteriorProbability > supportTau))
                ++numUnderflow;
        }

    if (isSparse && numUnderflow > 0)
        {
        size_t origN = n;

        TreeInfo* mapInfo = nullptr;
        double    bestPi  = -1.0;
        for (const auto& [h, info] : tcache)
            if (info && info->posteriorProbability > bestPi)
                { bestPi = info->posteriorProbability; mapInfo = info; }

        std::vector<char> visited(origN, 0);
        size_t numKept = 0;
        if (mapInfo && mapInfo->posteriorProbability > supportTau)
            {
            std::vector<TreeInfo*> stack;
            stack.push_back(mapInfo);
            visited[static_cast<size_t>(mapInfo->stateIndex)] = 1;
            numKept = 1;
            while (!stack.empty())
                {
                TreeInfo* cur = stack.back();
                stack.pop_back();
                for (TreeInfo* nb : cache->neighbors(cur))
                    {
                    if (!nb || nb->stateIndex < 0)
                        continue;
                    if (!(nb->posteriorProbability > supportTau))
                        continue;
                    char& v = visited[static_cast<size_t>(nb->stateIndex)];
                    if (!v)
                        { v = 1; ++numKept; stack.push_back(nb); }
                    }
                }
            }

        if (numKept < origN)
            {
            double massDropped = 0.0, massStranded = 0.0;
            size_t numStranded = 0;
            for (const auto& [h, info] : tcache)
                {
                if (!info || visited[static_cast<size_t>(info->stateIndex)])
                    continue;
                massDropped += info->posteriorProbability;
                if (info->posteriorProbability > supportTau)
                    { massStranded += info->posteriorProbability; ++numStranded; }
                }

            std::vector<uint64_t> newHashes;
            newHashes.reserve(numKept);
            hashToIndex.clear();
            hashToIndex.reserve(numKept * 2);
            Eigen::Index nidx = 0;
            for (const auto& [h, info] : tcache)
                {
                if (!info)
                    continue;
                if (visited[static_cast<size_t>(info->stateIndex)])
                    {
                    info->stateIndex = nidx;
                    hashToIndex[h] = nidx;
                    newHashes.push_back(h);
                    ++nidx;
                    }
                else
                    info->stateIndex = -1;
                }
            stateHashes.swap(newHashes);
            n = numKept;
            supportWasRestricted = true;

            std::cerr << "MarkovChainAnalyzer (" << name << "): dropped " << (origN - numKept)
                      << " of " << origN << " topologies whose posterior underflowed to zero ("
                      << massDropped << " posterior mass); analyzing the MAP tree's connected support of "
                      << numKept << " states.\n";
            if (massStranded > 1.0e-8)
                std::cerr << "Warning: " << numStranded << " topologies carrying " << massStranded
                          << " posterior mass form a component unreachable from the MAP tree except through "
                             "underflowed states; the chain is near-reducible and that mass is excluded from "
                             "the spectral analysis.\n";
            }
        }

    // ---- Number the states for locality ----
    // The order of the hash map holding the trees scatters each tree's neighbors across the whole state
    // vector, so every entry of a product with the kernel reads x from a random place in it. Numbered
    // breadth-first from the MAP tree over this move's neighbor graph, a tree's neighbors lie in its own
    // level or the next, and more of those reads hit cache: on the 10-taxon TBR kernel the eigensolves
    // and the Kemeny estimate ran about 30% faster. Every output carries the tree hash beside the state
    // index, so only the order of the per-state rows changes. Small (dense) state spaces keep the map's
    // order. States outside the support (stateIndex < 0) are neither numbered nor crossed.
    if (isSparse && n > 1)
        {
        size_t tableSize = cache->size();
        std::vector<int64_t> newIndex(tableSize, -1);            // by table index
        std::vector<uint32_t> order;                              // table indices in the new order
        order.reserve(n);
        TreeInfo* start = nullptr;
        double    bestPi = -1.0;
        for (const auto& [h, info] : tcache)
            if (info && info->stateIndex >= 0 && info->posteriorProbability > bestPi)
                { bestPi = info->posteriorProbability; start = info; }
        auto visit = [&](TreeInfo* root)
            {
            size_t head = order.size();
            newIndex[root->index] = static_cast<int64_t>(order.size());
            order.push_back(root->index);
            while (head < order.size())
                {
                NeighborList nbrs = cache->neighbors(cache->treeAt(order[head++]));
                for (uint32_t k = 0; k < nbrs.size(); ++k)
                    {
                    uint32_t j = nbrs.indexAt(k);
                    if (newIndex[j] >= 0)
                        continue;
                    TreeInfo* nb = cache->treeAt(j);
                    if (nb->stateIndex < 0)
                        continue;
                    newIndex[j] = static_cast<int64_t>(order.size());
                    order.push_back(j);
                    }
                }
            };
        if (start != nullptr)
            visit(start);
        for (const auto& [h, info] : tcache)                      // should the graph not be connected
            if (info && info->stateIndex >= 0 && newIndex[info->index] < 0)
                visit(info);

        // the new index of each tree in the map's order (the order of the former numbering)
        mapOrderState.clear();
        mapOrderState.reserve(n);
        for (const auto& [h, info] : tcache)
            if (info && info->stateIndex >= 0)
                mapOrderState.push_back(static_cast<Eigen::Index>(newIndex[info->index]));

        // The reachability summaries count the states reachable from, and able to reach, "state 0", which
        // was the first tree of the map; that tree stays their reference, so those columns keep their
        // meaning across the renumbering.
        reachabilityReference = mapOrderState.empty() ? 0 : mapOrderState[0];

        hashToIndex.clear();
        hashToIndex.reserve(n * 2);
        for (size_t k = 0; k < order.size(); ++k)
            {
            TreeInfo* info = cache->treeAt(order[k]);
            info->stateIndex = static_cast<int64_t>(k);
            stateHashes[k] = info->hash;
            hashToIndex[info->hash] = static_cast<Eigen::Index>(k);
            }
        }

    pi = Vector::Zero(static_cast<Eigen::Index>(n));

    // The kernel is written row by row, straight into its final compressed row-major storage. Earlier
    // versions gathered every edge as a (row, column, value) triplet, had Eigen sort them through a
    // transposed temporary into a column-major matrix, and then copied that into a row-major one for
    // the products: at ten taxa, for TBR, some 12 GB of triplets and 9 GB of temporary on the way to
    // two 7 GB copies of the same matrix. One row-major copy now serves every routine; those that
    // need a column of P read it from the corresponding row through detailed balance.
    std::vector<TreeInfo*> rowTree(n, nullptr);                   // the tree at each state index
    for (const auto& [h, info] : tcache)
        if (info && info->stateIndex >= 0)
            rowTree[static_cast<size_t>(info->stateIndex)] = info;

    // The transition probability from i to j requires the REVERSE proposal probability, q(i|j),
    // which is a property of j's neighborhood rather than of i's. Searching j's neighbor list for i
    // would cost O(degree) for every edge, making the whole construction O(n * degree^2) -- ruinous
    // for TBR, whose neighborhoods are twenty times larger than those of NNI.
    //
    // No search is needed, and no recomputation either. The proposal distribution of a tree j is
    //
    //     q(i|j) = exp[ (lnL_i - lnLmax_j) * beta ] / Z_j,
    //
    // in which lnLmax_j is the largest log likelihood among j's neighbors and Z_j the normalizing
    // constant. Both scalars were already formed when the proposal probabilities were cached, and
    // TreeCache keeps them, so the reverse proposal probability is available in constant time.
    if (cache->hasNeighborProposalProbabilities() == false)
        throw std::runtime_error("MarkovChainAnalyzer (" + name + "): proposal probabilities have not been cached");

    // Row ii of P as (column, P_ij) pairs, columns ascending, with the entries for a neighbor listed
    // more than once summed in the order they were listed (the order in which the old triplet
    // assembly summed them). Also returns the row's proposal-mass defect and its self transition.
    auto formRow = [&](size_t ii, std::vector<std::pair<int, double>>& row, double& defect, double& selfProb)
        {
        TreeInfo* info_from = rowTree[ii];
        double sumTransition = 0.0;

        NeighborList nbrs = cache->neighbors(info_from);
        double rawProposalMass = 0.0;
        for (uint32_t k = 0; k < nbrs.size(); ++k)
            rawProposalMass += cache->neighborProposalProbability(info_from, nbrs[k]);
        defect = std::abs(rawProposalMass - 1.0);

        double pi_i   = info_from->posteriorProbability;

        row.clear();
        for (uint32_t k = 0; k < nbrs.size(); ++k)
            {
            TreeInfo* info_to = nbrs[k];
            if (info_to->stateIndex < 0)
                continue;
            int j = static_cast<int>(info_to->stateIndex);

            double q_ij = cache->neighborProposalProbability(info_from, info_to);
            if (q_ij <= 0.0)
                continue;

            double q_ji = cache->neighborProposalProbability(info_to, info_from);

            double pi_j = info_to->posteriorProbability;
            if (pi_i <= 0.0 || q_ji <= 0.0)
                continue;

            double ratio      = (pi_j * q_ji) / (pi_i * q_ij);
            double alpha      = std::min(1.0, ratio);
            double trans_prob = q_ij * alpha;

            if (trans_prob > 0.0)
                {
                row.emplace_back(j, trans_prob);
                sumTransition += trans_prob;
                }
            }

        selfProb = 1.0 - sumTransition;
        if (selfProb < 0.0 && selfProb > -1e-12)
            selfProb = 0.0;
        row.emplace_back(static_cast<int>(ii), std::max(0.0, selfProb));

        std::stable_sort(row.begin(), row.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });

        // merge the repeated columns in place
        size_t out = 0;
        for (size_t k = 0; k < row.size(); )
            {
            int    j = row[k].first;
            double v = row[k].second;
            for (++k; k < row.size() && row[k].first == j; ++k)
                v += row[k].second;
            row[out++] = { j, v };
            }
        row.resize(out);
        };

    double worstProposalDefect = 0.0;
    std::vector<std::pair<Eigen::Index, double>> negativeSelf;    // (row, P_ii) for the warning

    if (isSparse)
        {
        // The rows are formed in parallel, twice: once to learn each row's length, from which the
        // compressed row offsets follow, and once to write the row into its place. Forming a row is a
        // few exponentials per entry, and done on one core this was some 35 s per TBR kernel at ten
        // taxa; the second pass costs less than holding a second copy of the kernel would in memory.
        const int parts = 256;
        std::vector<int>    rowLength(n, 0);
        std::vector<double> partDefect(static_cast<size_t>(parts), 0.0);
        std::vector<std::vector<std::pair<Eigen::Index, double>>> partNegative(static_cast<size_t>(parts));
        forRowRanges(threadPool, static_cast<Eigen::Index>(n), parts, [&](Eigen::Index lo, Eigen::Index hi, int part)
            {
            std::vector<std::pair<int, double>> row;
            double defect = 0.0, selfProb = 0.0;
            for (Eigen::Index i = lo; i < hi; ++i)
                {
                size_t ii = static_cast<size_t>(i);
                formRow(ii, row, defect, selfProb);
                rowLength[ii] = static_cast<int>(row.size());
                pi(i) = rowTree[ii]->posteriorProbability;
                partDefect[static_cast<size_t>(part)] = std::max(partDefect[static_cast<size_t>(part)], defect);
                if (selfProb < 0.0)
                    partNegative[static_cast<size_t>(part)].push_back({ i, selfProb });
                }
            });
        for (int part = 0; part < parts; ++part)
            {
            worstProposalDefect = std::max(worstProposalDefect, partDefect[static_cast<size_t>(part)]);
            for (const auto& e : partNegative[static_cast<size_t>(part)])
                negativeSelf.push_back(e);
            }

        size_t numEntries = 0;
        for (size_t ii = 0; ii < n; ++ii)
            numEntries += static_cast<size_t>(rowLength[ii]);
        if (numEntries > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("MarkovChainAnalyzer (" + name + "): kernel too large for 32-bit indices");

        P_row_sparse = RowSparseMatrix(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
        P_row_sparse.resizeNonZeros(static_cast<Eigen::Index>(numEntries));
        int* outer = P_row_sparse.outerIndexPtr();
        outer[0] = 0;
        for (size_t ii = 0; ii < n; ++ii)
            outer[ii + 1] = outer[ii] + rowLength[ii];
        std::vector<int>().swap(rowLength);
        int*    inner  = P_row_sparse.innerIndexPtr();
        double* values = P_row_sparse.valuePtr();

        forRowRanges(threadPool, static_cast<Eigen::Index>(n), parts, [&](Eigen::Index lo, Eigen::Index hi, int)
            {
            std::vector<std::pair<int, double>> row;
            double defect = 0.0, selfProb = 0.0;
            for (Eigen::Index i = lo; i < hi; ++i)
                {
                formRow(static_cast<size_t>(i), row, defect, selfProb);
                int at = outer[i];
                for (const auto& [j, v] : row)
                    {
                    inner[at]  = j;
                    values[at] = v;
                    ++at;
                    }
                }
            });
        }
    else
        {
        P_dense = DenseMatrix::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
        std::vector<std::pair<int, double>> row;
        for (size_t ii = 0; ii < n; ++ii)
            {
            Eigen::Index i = static_cast<Eigen::Index>(ii);
            double defect = 0.0, selfProb = 0.0;
            formRow(ii, row, defect, selfProb);
            pi(i) = rowTree[ii]->posteriorProbability;
            worstProposalDefect = std::max(worstProposalDefect, defect);
            if (selfProb < 0.0)
                negativeSelf.push_back({ i, selfProb });
            for (const auto& [j, v] : row)
                P_dense(i, j) += v;
            }
        }
    for (const auto& [i, v] : negativeSelf)
        std::cerr << "Warning: negative self transition at row " << i << ": " << v << "\n";

    if (worstProposalDefect > 1e-6)
        std::cerr << "Warning: proposal probabilities deviate from 1 by up to "
                  << worstProposalDefect << "; the acceptance-rate interpretation of "
                     "1 - P_ii assumes a normalized proposal with no explicit self-proposal.\n";

    denseEigenReady = false;
    clearSpectralCache();
}


bool MarkovChainAnalyzer::checkDetailedBalance(double tol) const {

    DetailedBalanceInfo info = computeDetailedBalanceInfo(tol);
    if (info.skipped) 
        {
        std::cout << "   * Detailed balance: skipped for large n (n = " << n << ")\n";
        return false;
        }
    std::cout << "   * Detailed balance max |pi_i P_ij - pi_j P_ji|: " << info.max_abs_error << "\n";
    std::cout << "   * Detailed balance max relative discrepancy: " << info.max_relative_error << "\n";
    std::cout << "   * Reversible by tolerance " << tol << ": " << (info.reversible ? "yes" : "no") << "\n";
    return info.reversible;
}

bool MarkovChainAnalyzer::checkIrreducible(double threshold, std::ostream& os) const {

    IrreducibilityInfo info = computeIrreducibilityInfo(threshold);
    os << "   * Irreducible transition graph: " << (info.irreducible ? "yes" : "no") << "\n";
    os << "   * Irreducibility threshold: " << info.threshold << "\n";
    os << "   * States reachable from state 0: " << info.states_reachable_from_0 << " / " << info.num_states << "\n";
    os << "   * States that can reach state 0: " << info.states_that_can_reach_0 << " / " << info.num_states << "\n";
    os << "   * Out-degree range, excluding self transitions: [" << info.min_out_degree << ", " << info.max_out_degree << "]\n";
    os << "   * Minimum probability of leaving a state: " << info.min_leave_probability << "\n";
    os << "   * States with zero positive-probability outgoing moves, excluding self transitions: " << info.num_states_with_zero_out_degree << "\n";
    os << "   * States with tiny leaving probability: " << info.num_states_with_tiny_leave_probability << "\n";
    return info.irreducible;
}

double MarkovChainAnalyzer::chiSquareDistanceFromStart(Eigen::Index x, int t) const {

    // chi^2_x(t) = sum_y (P^t(x,y) - pi_y)^2 / pi_y = (1/pi_x) sum_{j>=2} lambda_j^{2t} u_{xj}^2.
    if (x < 0 || x >= static_cast<Eigen::Index>(n))
        return std::numeric_limits<double>::quiet_NaN();

    Eigen::VectorXd evals;
    DenseMatrix U;
    if (!spectralModesForFunctionals(evals, U))
        return std::numeric_limits<double>::quiet_NaN();

    double chi = 0.0;
    for (Eigen::Index j = 0; j < evals.size(); ++j)
        {
        double lam = evals(j);
        if (std::abs(lam - 1.0) < 1e-8)
            continue;
        double uxj = U(x, j);
        chi += std::pow(lam, 2.0 * t) * uxj * uxj;
        }
    double px = pi(x);
    return px > 0.0 ? chi / px : std::numeric_limits<double>::quiet_NaN();
}

MarkovChainAnalyzer::Vector MarkovChainAnalyzer::chiSquareDistanceCurve(Eigen::Index x, const std::vector<int>& times) const {

    Vector out;
    if (x < 0 || x >= static_cast<Eigen::Index>(n) || times.empty())
        return out;

    Eigen::VectorXd evals;
    DenseMatrix U;
    if (!spectralModesForFunctionals(evals, U))
        return out;

    double px = pi(x);
    if (!(px > 0.0))
        return out;

    out.resize(static_cast<Eigen::Index>(times.size()));
    for (size_t ti = 0; ti < times.size(); ++ti)
        {
        double chi = 0.0;
        int t = times[ti];
        for (Eigen::Index j = 0; j < evals.size(); ++j)
            {
            double lam = evals(j);
            if (std::abs(lam - 1.0) < 1e-8)
                continue;
            double uxj = U(x, j);
            chi += std::pow(lam, 2.0 * t) * uxj * uxj;
            }
        out(static_cast<Eigen::Index>(ti)) = chi / px;
        }
    return out;
}

void MarkovChainAnalyzer::clearSpectralCache(void) const {

    spectralCacheValid = false;
    spectralCacheNev = 0;
    spectralCache = SpectralInfo();
    slowModeValues.resize(0);
    slowModeVectors.resize(0, 0);
}

double MarkovChainAnalyzer::conductanceForSet(const std::vector<Eigen::Index>& set) const {

    if (set.empty() || set.size() >= n)
        return std::numeric_limits<double>::quiet_NaN();

    Eigen::Index N = static_cast<Eigen::Index>(n);
    std::vector<char> inSet(static_cast<size_t>(N), 0);
    for (Eigen::Index idx : set)
        if (idx >= 0 && idx < N)
            inSet[static_cast<size_t>(idx)] = 1;

    double piA = 0.0;
    for (Eigen::Index i = 0; i < N; i++)
        if (inSet[static_cast<size_t>(i)])
            piA += pi(i);
    double denom = std::min(piA, 1.0 - piA);
    if (denom <= 0.0)
        return std::numeric_limits<double>::quiet_NaN();

    double flow = 0.0;
    if (isSparse) 
        {
        ensureRowSparse();
        for (Eigen::Index i = 0; i < N; i++) 
            {
            if (!inSet[static_cast<size_t>(i)]) continue;
            for (RowSparseMatrix::InnerIterator it(P_row_sparse, i); it; ++it)
                if (!inSet[static_cast<size_t>(it.col())])
                    flow += pi(i) * it.value();
            }
        } 
    else 
        {
        for (Eigen::Index i = 0; i < N; ++i) 
            {
            if (!inSet[static_cast<size_t>(i)]) continue;
            for (Eigen::Index j = 0; j < N; ++j)
                if (!inSet[static_cast<size_t>(j)])
                    flow += pi(i) * P_dense(i, j);
            }
        }
    return flow / denom;
}

DetailedBalanceInfo MarkovChainAnalyzer::computeDetailedBalanceInfo(double tol) const {

    DetailedBalanceInfo info;
    if (n > detailedBalanceStateLimit) 
        {
        info.skipped = true;
        return info;
        }

    Eigen::Index N = static_cast<Eigen::Index>(n);
    if (isSparse) 
        {
        ensureRowSparse();
        for (Eigen::Index i = 0; i < N; ++i) 
            {
            for (RowSparseMatrix::InnerIterator it(P_row_sparse, i); it; ++it) 
                {
                Eigen::Index j = it.col();
                double a = pi(i) * it.value();
                double b = pi(j) * P_row_sparse.coeff(j, i);
                double d = std::abs(a - b);
                double denom = std::max({std::abs(a), std::abs(b), std::numeric_limits<double>::min()});
                info.max_abs_error = std::max(info.max_abs_error, d);
                info.sum_abs_error += d;
                info.max_relative_error = std::max(info.max_relative_error, d / denom);
                }
            }
        } 
    else 
        {
        for (Eigen::Index i = 0; i < N; ++i) 
            {
            for (Eigen::Index j = 0; j < N; ++j) 
                {
                double a = pi(i) * P_dense(i, j);
                double b = pi(j) * P_dense(j, i);
                double d = std::abs(a - b);
                double denom = std::max({std::abs(a), std::abs(b), std::numeric_limits<double>::min()});
                info.max_abs_error = std::max(info.max_abs_error, d);
                info.sum_abs_error += d;
                info.max_relative_error = std::max(info.max_relative_error, d / denom);
                }
            }
        }
    info.reversible = (info.max_abs_error < tol);
    return info;
}

IrreducibilityInfo MarkovChainAnalyzer::computeIrreducibilityInfo(double threshold, double tinyLeaveProb) const {

    IrreducibilityInfo info;
    info.threshold = threshold;
    info.num_states = n;
    if (n == 0)
        return info;

    Eigen::Index N = static_cast<Eigen::Index>(n);
    info.min_out_degree = std::numeric_limits<size_t>::max();
    info.max_out_degree = 0;
    info.min_leave_probability = std::numeric_limits<double>::infinity();

    if (isSparse) 
        {
        std::vector<IrreducibilityInfo> infos(1, info);
        outDegreeStats({ threshold }, tinyLeaveProb, infos);
        info = infos[0];
        }
    else 
        {
        for (Eigen::Index i = 0; i < N; ++i) 
            {
            size_t outDegree = 0;
            double leaveProb = 0.0;
            for (Eigen::Index j = 0; j < N; ++j) 
                {
                double p = P_dense(i, j);
                if (j != i && p > threshold) 
                    {
                    ++outDegree;
                    leaveProb += p;
                    }
                }
            info.min_out_degree = std::min(info.min_out_degree, outDegree);
            info.max_out_degree = std::max(info.max_out_degree, outDegree);
            info.min_leave_probability = std::min(info.min_leave_probability, leaveProb);
            if (outDegree == 0)
                ++info.num_states_with_zero_out_degree;
            if (leaveProb <= tinyLeaveProb)
                ++info.num_states_with_tiny_leave_probability;
            }
        }

    if (info.min_out_degree == std::numeric_limits<size_t>::max())
        info.min_out_degree = 0;
    if (!std::isfinite(info.min_leave_probability))
        info.min_leave_probability = std::numeric_limits<double>::quiet_NaN();

    info.states_reachable_from_0 = countReachableForward(reachabilityReference, threshold);
    info.states_that_can_reach_0 = countReachableReverse(reachabilityReference, threshold);
    info.irreducible = (info.states_reachable_from_0 == n && info.states_that_can_reach_0 == n);
    return info;
}

// Preconditioned conjugate gradient for a symmetric positive-definite operator supplied as a
// function. Jacobi preconditioning by the diagonal of the operator is used, which matters because
// a sticky chain -- one whose proposals are usually rejected -- has a badly scaled diagonal, and
// the unpreconditioned iteration then converges slowly.
bool MarkovChainAnalyzer::conjugateGradient(const std::function<void(const Vector&, Vector&)>& applyA,
                                            const Vector& diagonal, const Vector& b, Vector& x,
                                            double tolerance, int maxIterations,
                                            int& iterations, double& relativeResidual) const {

    Eigen::Index N = b.size();
    x = Vector::Zero(N);

    using CgClock = std::chrono::steady_clock;
    const CgClock::time_point cgDeadline =
        CgClock::now() + std::chrono::duration_cast<CgClock::duration>(
                             std::chrono::duration<double>(s_solveBudgetSeconds));

    double bNorm = b.norm();
    if (bNorm == 0.0)
        {
        iterations       = 0;
        relativeResidual = 0.0;
        return true;
        }

    Vector r  = b;                                                // r = b - A x, with x = 0
    Vector z  = r.array() / diagonal.array();
    Vector p  = z;
    Vector Ap = Vector::Zero(N);
    double rz = r.dot(z);

    // The vector work of an iteration is done in three threaded sweeps -- p.Ap; the updates of x and r
    // with the preconditioned residual z and the sums r.r and r.z; the new direction p -- each over a
    // fixed partition of the rows with the partial sums combined in order. Done with Eigen it was some
    // seven serial passes, which on an NNI kernel of ten taxa cost more than the product with A. Below a
    // million states a sweep is too short to repay handing it to the pool, and the same partition is
    // walked serially, with the same result.
    const int parts = 64;
    ThreadPool* pool = (N >= (Eigen::Index(1) << 20)) ? threadPool : nullptr;
    std::vector<double> partA(static_cast<size_t>(parts)), partB(static_cast<size_t>(parts));
    double* xp = x.data();
    double* rp = r.data();
    double* zp = z.data();
    double* pp = p.data();
    const double* ap = Ap.data();
    const double* dp = diagonal.data();
    double rr = r.squaredNorm();

    for (iterations = 0; iterations < maxIterations; ++iterations)
        {
        if ((iterations & 255) == 0 && CgClock::now() >= cgDeadline)
            break;                                                // budget spent -> return not-converged
        applyA(p, Ap);
        forRowRanges(pool, N, parts, [&](Eigen::Index lo, Eigen::Index hi, int part)
            {
            double acc[4] = { 0.0, 0.0, 0.0, 0.0 };              // four chains, so the adds overlap
            Eigen::Index i = lo;
            for (; i + 4 <= hi; i += 4)
                for (int k = 0; k < 4; ++k)
                    acc[k] += pp[i + k] * ap[i + k];
            for (; i < hi; ++i)
                acc[0] += pp[i] * ap[i];
            partA[static_cast<size_t>(part)] = (acc[0] + acc[1]) + (acc[2] + acc[3]);
            });
        double pAp = 0.0;
        for (double v : partA)
            pAp += v;
        if (!(pAp > 0.0))
            break;                                                // positive definiteness lost
        double alpha = rz / pAp;
        forRowRanges(pool, N, parts, [&](Eigen::Index lo, Eigen::Index hi, int part)
            {
            double accRR[4] = { 0.0, 0.0, 0.0, 0.0 }, accRZ[4] = { 0.0, 0.0, 0.0, 0.0 };
            for (Eigen::Index i = lo; i < hi; ++i)
                {
                xp[i] += alpha * pp[i];
                double ri = rp[i] - alpha * ap[i];
                rp[i] = ri;
                double zi = ri / dp[i];
                zp[i] = zi;
                accRR[i & 3] += ri * ri;
                accRZ[i & 3] += ri * zi;
                }
            partA[static_cast<size_t>(part)] = (accRR[0] + accRR[1]) + (accRR[2] + accRR[3]);
            partB[static_cast<size_t>(part)] = (accRZ[0] + accRZ[1]) + (accRZ[2] + accRZ[3]);
            });
        rr = 0.0;
        double rzNext = 0.0;
        for (int part = 0; part < parts; ++part)
            {
            rr     += partA[static_cast<size_t>(part)];
            rzNext += partB[static_cast<size_t>(part)];
            }

        relativeResidual = std::sqrt(rr) / bNorm;
        if (relativeResidual < tolerance)
            {
            ++iterations;
            return true;
            }

        double beta = rzNext / rz;
        forRowRanges(pool, N, parts, [&](Eigen::Index lo, Eigen::Index hi, int)
            {
            for (Eigen::Index i = lo; i < hi; ++i)
                pp[i] = zp[i] + beta * pp[i];
            });
        rz = rzNext;
        }

    relativeResidual = std::sqrt(rr) / bNorm;
    return relativeResidual < tolerance;
}

// The out-degree and leaving-probability summaries of computeIrreducibilityInfo for several
// thresholds in one threaded pass over the kernel, each row's leaving probability summed in column
// order as before. infos[t] must arrive initialized (threshold, num_states, the min/max sentinels).
void MarkovChainAnalyzer::outDegreeStats(const std::vector<double>& thresholds, double tinyLeaveProb,
                                         std::vector<IrreducibilityInfo>& infos) const {

    ensureRowSparse();
    const size_t T = thresholds.size();
    const int parts = 256;
    struct PartStats {
        std::vector<size_t> minDeg, maxDeg, zeroDeg, tinyLeave;
        std::vector<double> minLeave;
    };
    std::vector<PartStats> ps(static_cast<size_t>(parts));
    for (PartStats& p : ps)
        {
        p.minDeg.assign(T, std::numeric_limits<size_t>::max());
        p.maxDeg.assign(T, 0);
        p.zeroDeg.assign(T, 0);
        p.tinyLeave.assign(T, 0);
        p.minLeave.assign(T, std::numeric_limits<double>::infinity());
        }
    forRowRanges(threadPool, static_cast<Eigen::Index>(n), parts, [&](Eigen::Index lo, Eigen::Index hi, int part)
        {
        PartStats& p = ps[static_cast<size_t>(part)];
        std::vector<size_t> outDegree(T);
        std::vector<double> leaveProb(T);
        for (Eigen::Index i = lo; i < hi; ++i)
            {
            std::fill(outDegree.begin(), outDegree.end(), 0);
            std::fill(leaveProb.begin(), leaveProb.end(), 0.0);
            for (RowSparseMatrix::InnerIterator it(P_row_sparse, i); it; ++it)
                {
                if (it.col() == i)
                    continue;
                double v = it.value();
                for (size_t t = 0; t < T; ++t)
                    if (v > thresholds[t])
                        {
                        ++outDegree[t];
                        leaveProb[t] += v;
                        }
                }
            for (size_t t = 0; t < T; ++t)
                {
                p.minDeg[t]   = std::min(p.minDeg[t], outDegree[t]);
                p.maxDeg[t]   = std::max(p.maxDeg[t], outDegree[t]);
                p.minLeave[t] = std::min(p.minLeave[t], leaveProb[t]);
                if (outDegree[t] == 0)
                    ++p.zeroDeg[t];
                if (leaveProb[t] <= tinyLeaveProb)
                    ++p.tinyLeave[t];
                }
            }
        });
    for (size_t t = 0; t < T; ++t)
        for (const PartStats& p : ps)
            {
            infos[t].min_out_degree = std::min(infos[t].min_out_degree, p.minDeg[t]);
            infos[t].max_out_degree = std::max(infos[t].max_out_degree, p.maxDeg[t]);
            infos[t].min_leave_probability = std::min(infos[t].min_leave_probability, p.minLeave[t]);
            infos[t].num_states_with_zero_out_degree += p.zeroDeg[t];
            infos[t].num_states_with_tiny_leave_probability += p.tinyLeave[t];
            }
}

// The number of states reachable from start along transitions above the threshold (or, reversed,
// the number that can reach it), by a breadth-first search run level by level over the thread pool.
// A state is claimed by an atomic exchange on its flag, so each is counted once whatever the order
// in which the threads reach it. At ten taxa a TBR kernel has some 590 million entries, and the
// single-threaded depth-first search, run forward and backward for each of seven thresholds, took
// longer than the eigensolve. Reversed, the successors of i are the j with P(j,i) above the
// threshold: column i of P. The kernel is stored by rows, but it is reversible, so
// P(j,i) = pi_i P(i,j) / pi_j, and row i, whose support is the same, gives the column.
size_t MarkovChainAnalyzer::countReachableSparse(Eigen::Index start, double threshold, bool reverse) const {

    ensureRowSparse();
    Eigen::Index N = static_cast<Eigen::Index>(n);
    std::vector<unsigned char> visited(static_cast<size_t>(N), 0);
    visited[static_cast<size_t>(start)] = 1;
    std::vector<int> frontier(1, static_cast<int>(start));
    size_t count = 1;

    const int parts = (threadPool == nullptr) ? 1 : std::max(1, threadPool->getThreadCount() * 4);
    std::vector<std::vector<int>> next(static_cast<size_t>(parts));
    while (!frontier.empty())
        {
        int used = (frontier.size() < 4096) ? 1 : parts;
        forRowRanges(used == 1 ? nullptr : threadPool, static_cast<Eigen::Index>(frontier.size()), used,
                     [&](Eigen::Index lo, Eigen::Index hi, int part)
            {
            std::vector<int>& out = next[static_cast<size_t>(part)];
            out.clear();
            for (Eigen::Index f = lo; f < hi; ++f)
                {
                Eigen::Index i = frontier[static_cast<size_t>(f)];
                for (RowSparseMatrix::InnerIterator it(P_row_sparse, i); it; ++it)
                    {
                    Eigen::Index j = it.col();
                    if (j == i)
                        continue;
                    double v = reverse ? pi(i) * it.value() / pi(j) : it.value();
                    if (v <= threshold)
                        continue;
                    unsigned char* flag = &visited[static_cast<size_t>(j)];
                    if (__atomic_load_n(flag, __ATOMIC_RELAXED) == 0 &&
                        __atomic_exchange_n(flag, static_cast<unsigned char>(1), __ATOMIC_RELAXED) == 0)
                        out.push_back(static_cast<int>(j));
                    }
                }
            });
        frontier.clear();
        for (int part = 0; part < used; ++part)
            {
            frontier.insert(frontier.end(), next[static_cast<size_t>(part)].begin(), next[static_cast<size_t>(part)].end());
            count += next[static_cast<size_t>(part)].size();
            }
        }
    return count;
}

size_t MarkovChainAnalyzer::countReachableForward(Eigen::Index start, double threshold) const {

    Eigen::Index N = static_cast<Eigen::Index>(n);
    if (start < 0 || start >= N)
        return 0;

    if (isSparse)
        return countReachableSparse(start, threshold, false);

    std::vector<char> visited(static_cast<size_t>(N), 0);
    std::vector<Eigen::Index> stack;
    stack.reserve(1024);
    visited[static_cast<size_t>(start)] = 1;
    stack.push_back(start);
    size_t count = 1;

    while (!stack.empty()) 
        {
        Eigen::Index i = stack.back();
        stack.pop_back();
        for (Eigen::Index j = 0; j < N; ++j) 
            {
            if (j == i) continue;
            if (P_dense(i, j) <= threshold) continue;
            size_t jj = static_cast<size_t>(j);
            if (!visited[jj]) 
                {
                visited[jj] = 1;
                stack.push_back(j);
                ++count;
                }
            }
        }
    return count;
}

size_t MarkovChainAnalyzer::countReachableReverse(Eigen::Index start, double threshold) const {

    Eigen::Index N = static_cast<Eigen::Index>(n);
    if (start < 0 || start >= N)
        return 0;

    if (isSparse)
        return countReachableSparse(start, threshold, true);

    std::vector<char> visited(static_cast<size_t>(N), 0);
    std::vector<Eigen::Index> stack;
    stack.reserve(1024);
    visited[static_cast<size_t>(start)] = 1;
    stack.push_back(start);
    size_t count = 1;

    while (!stack.empty()) 
        {
        Eigen::Index i = stack.back();
        stack.pop_back();
        for (Eigen::Index j = 0; j < N; ++j) 
            {
            if (j == i) continue;
            if (P_dense(j, i) <= threshold) continue;
            size_t jj = static_cast<size_t>(j);
            if (!visited[jj]) 
                {
                visited[jj] = 1;
                stack.push_back(j);
                ++count;
                }
            }
        }
    return count;
}

MarkovChainAnalyzer::DenseMatrix MarkovChainAnalyzer::computeHittingTimes(void) const {

    if (n > denseStateLimit) 
        {
        std::cerr << "Skipping all-pairs hitting times for n = " << n << " because this is dense O(n^3) work.\n";
        return DenseMatrix();
        }

    Eigen::Index N = static_cast<Eigen::Index>(n);
    DenseMatrix Pmat = isSparse ? DenseMatrix(P_row_sparse) : P_dense;
    DenseMatrix I = DenseMatrix::Identity(N, N);
    DenseMatrix A = I - Pmat;
    for (Eigen::Index i = 0; i < N; i++)
        A.row(i) += pi.transpose();

    DenseMatrix Z = A.inverse();
    DenseMatrix M = DenseMatrix::Zero(N, N);
    for (Eigen::Index i = 0; i < N; i++)
        for (Eigen::Index j = 0; j < N; j++)
            M(i, j) = (i == j) ? 0.0 : (Z(j, j) - Z(i, j)) / pi(j);
    return M;
}

SpectralInfo MarkovChainAnalyzer::computeSpectralInfo(void) const {

    if (n > denseStateLimit) 
        {
        if (isSparse)
            return computeSpectralInfoSparse(defaultSparseEigenvalues);
        std::cerr << "Skipping dense eigensystem for n = " << n << " because it exceeds denseStateLimit.\n";
        return SpectralInfo();
        }

    SpectralInfo info;
    info.computed_sparse = false;

    if (n < 2)
        {
        info.spectral_status = "too_few_states";
        return info;
        }

    // Reversible chain => symmetrize and use a self-adjoint solver: real spectrum, no
    // spurious complex eigenvalues, and a full Euclidean-orthonormal eigenbasis of S
    // (whose columns map to pi-orthonormal eigenvectors of P).
    if (!ensureDenseEigensystem())
        {
        info.spectral_status = "dense_selfadjoint_failed";
        return info;
        }

    info.eigenvalues_real = denseEigenvalues;             // descending algebraic order
    info.n_converged = static_cast<int>(info.eigenvalues_real.size());
    finalizeSpectralReal(info);

    info.lambda1_error = std::abs(info.eigenvalues_real(0) - 1.0);
    info.spectral_valid = (info.eigenvalues_real.size() >= 2 &&
                           std::isfinite(info.lambda1_error) &&
                           info.lambda1_error < 1e-8 &&
                           std::isfinite(info.lambda2_abs) &&
                           info.lambda2_abs <= 1.0 + 1e-10);
    info.spectral_status = info.spectral_valid ? "ok_dense" : "invalid_dense";
    return info;
}

SpectralInfo MarkovChainAnalyzer::computeSpectralInfoSparse(int nev) const {

    if (!isSparse || n <= denseStateLimit)
        return computeSpectralInfo();

    SpectralInfo bestInfo;
    bestInfo.computed_sparse = true;
    bestInfo.spectral_status = "not_run";

    if (n < 2) 
        {
        bestInfo.spectral_status = "too_few_states";
        return bestInfo;
        }

    nev = std::max(3, nev);                       // need >= {1, lambda_2} at the top
    nev = std::min(nev, static_cast<int>(n) - 1);

    using Op = AnalyzerSymOp;
    Op op(this, static_cast<Eigen::Index>(n));

    // Guard the operator before handing it to Lanczos. A well-formed symmetrized kernel is a contraction
    // (spectral radius 1), so || S x || <= || x ||. A non-finite or hugely amplified product means a
    // posterior underflowed and left an unbounded 1/sqrt(pi) scaling on a state the support restriction
    // did not remove. Catch it here with an explanatory message rather than letting it surface later as
    // the opaque "TridiagEigen: eigen decomposition failed" from deep inside the eigensolver. The probe
    // also triggers the lazy Accelerate build (and its self-check) once, up front.
    {
        Eigen::VectorXd probe = Eigen::VectorXd::Random(static_cast<Eigen::Index>(n));
        double pn = probe.norm();
        if (pn > 0.0) probe /= pn;
        Eigen::VectorXd out(static_cast<Eigen::Index>(n));
        op.perform_op(probe.data(), out.data());
        double on = out.norm();
        if (!out.allFinite() || on > 1.0e3)
            throw std::runtime_error(
                "MarkovChainAnalyzer (" + name + "): the symmetrized kernel is not analyzable ("
                + (out.allFinite() ? ("|| S x || / || x || = " + std::to_string(on) + ", far above the bound of 1")
                                   : std::string("the product is non-finite"))
                + "). This is the signature of posterior underflow: a topology whose posterior fell "
                  "below the smallest representable double leaves an unbounded 1/sqrt(pi) in the "
                  "symmetrization. The connected-support restriction removes such states before this "
                  "point, so reaching here means one survived within the MAP tree's component -- the "
                  "log-likelihood spread of this data set likely exceeds what double precision can hold "
                  "across the reachable state space.");
    }

    // Deflate the known top eigenpair lambda_1 = 1, eigenvector proportional to sqrt(pi). Its residual
    // is checked directly (no iteration), and the eigensolver is run on the deflated operator so it
    // returns lambda_2 as its largest eigenvalue -- avoiding the exact-1-in-a-near-1-cluster situation
    // that stalls an undeflated solve on collapsed, multi-basin landscapes.
    Eigen::VectorXd w = pi.cwiseSqrt();
    double wnorm = w.norm();
    if (wnorm > 0.0)
        w /= wnorm;
    Eigen::VectorXd Sw(w.size());
    op.perform_op(w.data(), Sw.data());
    double lambda1Residual = (Sw - w).norm();          // || S w - 1 * w ||

    DeflatedSymOp dop(op, w);

    int kBot = 1;

    // Wall-clock cap for the whole solve. A pathological spectrum should cost a bounded amount of time
    // and be reported "solver_timed_out", not stall the run.
    using SolveClock = std::chrono::steady_clock;
    const SolveClock::time_point solveDeadline =
        SolveClock::now() + std::chrono::duration_cast<SolveClock::duration>(
                                std::chrono::duration<double>(s_solveBudgetSeconds));
    bool timedOut = false;

    auto computeCapped = [&](auto& eigs, Spectra::SortRule rule, int maxit, double tol, const char* which,
                             SolveClock::time_point deadline, bool& expired) -> int
        {
        // One uninterrupted Spectra solve, cut off from inside by the operator's deadline (see
        // AnalyzerSymOp). Earlier versions drove the solve in batches of 200 restarts and checked the
        // clock between them; a batch is a fraction of a second on a small kernel but some 30 minutes on a
        // 10-taxon TBR kernel, where it let a capped solve overrun fourfold. Smaller batches would bound
        // the overrun, but each re-entry rebuilds the Krylov basis, and in tests that sent the bottom solve
        // to an interior eigenvalue. Returns the number of converged eigenvalues (0 if cut off).
        auto t0 = SolveClock::now();
        int nconv = 0;
        bool converged = false;
        op.setDeadline(&deadline);
        try
            {
            eigs.init();
            nconv = (int)eigs.compute(rule, maxit, tol);
            converged = (eigs.info() == Spectra::CompInfo::Successful);
            }
        catch (const SolveBudgetSpent&)
            {
            expired = true;
            nconv = 0;
            }
        op.setDeadline(nullptr);
        double secs = std::chrono::duration<double>(SolveClock::now() - t0).count();
        std::cerr << "   [" << name << " " << which << "] " << (expired ? std::string("cut off") :
                                                                  std::to_string(eigs.num_iterations()) + " restarts")
                  << ", " << secs << " s, converged=" << converged
                  << (expired ? " (budget spent)" : "") << "\n";
        return nconv;
        };

    // ---- The slow end: lambda_2, lambda_3, ... ----
    // A plain Lanczos solve of the deflated operator is tried first, with a modest restart budget: when
    // the gap below 1 is comfortable it converges in a few dozen restarts and is the cheapest route.
    // When the posterior is split into weakly connected basins, lambda_2 sits in a tight cluster just
    // below 1 (1 - 6.5e-8 on a 9-taxon NNI kernel) that no affordable Krylov subspace resolves;
    // widening the subspace, as earlier versions did, only made each restart dominated by the serial
    // reorthogonalization, and such cells ran out the budget and reported NaN. Those cells are handed
    // to shift-invert Lanczos on (I - S + w w')^{-1}, which separates the cluster and converges in
    // seconds. A cluster that is degenerate but not close to 1 (exactly repeated eigenvalues, as a
    // uniform proposal on a symmetric neighborhood graph produces) is not separated by the inversion; the
    // wide-subspace Lanczos attempts of earlier versions are kept after shift-invert for those. If no
    // attempt converges fully, converged eigenpairs from a partial solve are used, flagged "ok_partial".
    // Whatever their source, the eigenpairs are checked against S itself below.
    // The plain solve either converges quickly or not at all: on every 9- and 10-taxon kernel tried it
    // converged within 17 restarts or ran out its budget, which was 100 restarts -- some 40 s on a
    // 10-taxon NNI kernel and minutes on a TBR one -- before shift-invert was tried. 40 keeps a wide
    // margin over the slowest success.
    const int    standardTopRestarts = 40;       // ~640 kernel products
    const double standardTol         = 1e-10;
    int ncvTop = std::min(static_cast<int>(n), std::max(std::max(2 * nev + 1, 20), nev + 2));
    int    ncvUsed   = ncvTop;
    int    maxitUsed = standardTopRestarts;
    double tolUsed   = standardTol;

    std::vector<std::pair<double, Eigen::VectorXd>> topPairs;
    bool sawNaN = false;
    int  nconvTop = 0;
    auto collectTop = [&](const Eigen::VectorXd& vals, const Eigen::MatrixXd& vecs)
        {
        topPairs.clear();
        sawNaN = false;
        nconvTop = static_cast<int>(vals.size());
        for (Eigen::Index i = 0; i < vals.size(); ++i)
            {
            if (!std::isfinite(vals(i))) { sawNaN = true; continue; }
            // Skip the deflated unit mode: its eigenvector is w, which is not an eigenvector of the
            // deflated operator at eigenvalue 1. Genuine eigenvectors (lambda != 1) are orthogonal to w,
            // so a large overlap identifies the deflated direction.
            if (std::abs(w.dot(vecs.col(i))) > 0.5)
                continue;
            topPairs.push_back({vals(i), vecs.col(i)});
            }
        };

    int  attemptNumber = 0;
    bool topOK = false;
    bool usedShiftInvert = false;

    // The converged eigenpairs of the best attempt that did not converge fully, as a last resort.
    Eigen::VectorXd partialVals;
    DenseMatrix     partialVecs;
    auto keepPartial = [&](const Eigen::VectorXd& vals, const DenseMatrix& vecs, int ncv, int maxit, double tol)
        {
        if (vals.size() > partialVals.size())
            {
            partialVals = vals;
            partialVecs = vecs;
            ncvUsed = ncv; maxitUsed = maxit; tolUsed = tol;
            }
        };

    auto runStandard = [&](int ncv, int maxit, double tol, const char* which) -> bool
        {
        ++attemptNumber;
        ncv = std::min(static_cast<int>(n), std::max(ncv, nev + 2));
        try
            {
            Spectra::SymEigsSolver<DeflatedSymOp> eigsTop(dop, nev, ncv);
            computeCapped(eigsTop, Spectra::SortRule::LargestAlge, maxit, tol, which, solveDeadline, timedOut);
            if (timedOut)
                return false;                                         // cut off mid-solve: no usable state
            Eigen::VectorXd vals = eigsTop.eigenvalues();             // the converged Ritz values only
            if (vals.size() == 0)
                return false;
            if (eigsTop.info() != Spectra::CompInfo::Successful)
                {
                keepPartial(vals, eigsTop.eigenvectors(), ncv, maxit, tol);
                return false;
                }
            collectTop(vals, eigsTop.eigenvectors());
            if (sawNaN || topPairs.empty())
                return false;
            ncvUsed = ncv; maxitUsed = maxit; tolUsed = tol;
            return true;
            }
        catch (const std::exception& e)
            {
            std::cerr << "Note: Lanczos solve of the slow modes for " << name << " failed internally ("
                      << e.what() << ").\n";
            }
        return false;
        };

    auto runShiftInvert = [&](void) -> bool
        {
        ++attemptNumber;
        usedShiftInvert = true;
        Eigen::VectorXd lam;
        DenseMatrix U;
        const int siRestarts = 50;
        int siNcv = std::min(static_cast<int>(n), 2 * nev + 8);
        if (!shiftInvertTopModes(nev, w, siRestarts, solveDeadline, timedOut, lam, U))
            {
            if (lam.size() > 0 && !timedOut)
                keepPartial(lam, U, siNcv, siRestarts, 1e-9);
            return false;
            }
        collectTop(lam, U);
        if (sawNaN || topPairs.empty())
            return false;
        ncvUsed = siNcv; maxitUsed = siRestarts; tolUsed = 1e-9;
        return true;
        };

    topOK = runStandard(ncvTop, standardTopRestarts, standardTol, "top");
    if (!topOK && !timedOut)
        topOK = runShiftInvert();
    if (!topOK && !timedOut)
        topOK = runStandard(std::max(6 * nev + 40, 120), 1000, 1e-11, "top, wide");
    if (!topOK && !timedOut)
        topOK = runStandard(std::max(12 * nev + 80, 300), 2000, 1e-12, "top, wider");

    bool partialTop = false;
    if (!topOK && !timedOut && partialVals.size() > 0)
        {
        collectTop(partialVals, partialVecs);
        topOK = partialTop = (!sawNaN && !topPairs.empty());
        if (partialTop)
            std::cerr << "Note: the slow-mode solve for " << name << " converged " << partialVals.size()
                      << " of " << nev << " eigenpairs; using those (checked against S below).\n";
        }

    // ---- The fast end: lambda_min, from the undeflated operator ----
    // Only lambda_min itself is wanted (for the SLEM and the lambda_min column), so one eigenpair is
    // requested. Earlier versions asked for two, and where the bottom of the spectrum is a repeated
    // eigenvalue a single-vector Lanczos solve sees only one copy of it, so the second requested value
    // lay in the continuum above and never converged: a 9-taxon NNI2 kernel ran some 380 s at ncv = 120
    // without finishing. The solve runs on S + 2I (see ShiftedSymOp), so its tolerance of 1e-7 is in effect
    // an absolute one on lambda_min rather than one relative to |lambda_min|, which near zero was out of
    // reach. It starts at ncv = 60: where lambda_min lies in the cluster near zero a 20-vector subspace
    // spent its 300 restarts without converging before the wider one, which converges in a few dozen,
    // was tried; elsewhere the wider subspace converges in a handful of restarts.
    //
    // The bottom solve also has its own wall-clock cap, a quarter of the budget. Where lambda_min sits in
    // the dense bulk of the spectrum near zero it converges very slowly: a 10-taxon TBR kernel ran 201
    // restarts (1,880 s) without converging, and the timeout then discarded a lambda_2 that had converged
    // in 10 s. lambda_2, the worst-case IACT, and the slow modes do not depend on lambda_min, so when the
    // bottom solve runs out they are kept, and only lambda_min and the quantities built on the SLEM (the
    // spectral gap, relaxation time, and mixing-time bound) are reported as unresolved.
    const double bottomTol = 1e-7;
    const SolveClock::time_point bottomDeadline =
        std::min(solveDeadline, SolveClock::now() + std::chrono::duration_cast<SolveClock::duration>(
                                    std::chrono::duration<double>(0.25 * s_solveBudgetSeconds)));
    bool bottomTimedOut = false;
    struct BottomAttempt { int ncv; int maxit; double tol; };
    const BottomAttempt bottomAttempts[] = { { 60, 300, bottomTol },
                                             { 120, 300, bottomTol } };
    std::vector<std::pair<double, Eigen::VectorXd>> botPairs;
    int nconvBot = 0;
    const double bottomShift = 2.0;
    ShiftedSymOp sop(op, bottomShift);
    for (const BottomAttempt& attempt : bottomAttempts)
        {
        if (timedOut || bottomTimedOut || !topOK)
            break;
        int ncvBot = std::min(static_cast<int>(n), std::max(attempt.ncv, kBot + 2));
        try
            {
            Spectra::SymEigsSolver<ShiftedSymOp> eigsBot(sop, kBot, ncvBot);
            nconvBot = computeCapped(eigsBot, Spectra::SortRule::SmallestAlge, attempt.maxit, attempt.tol, "bot",
                                     bottomDeadline, bottomTimedOut);
            if (nconvBot > 0 && eigsBot.info() == Spectra::CompInfo::Successful)
                {
                Eigen::VectorXd vb = eigsBot.eigenvalues();
                Eigen::MatrixXd Vb = eigsBot.eigenvectors();
                botPairs.clear();
                bool nanBot = false;
                for (Eigen::Index i = 0; i < vb.size(); ++i)
                    {
                    if (!std::isfinite(vb(i))) { nanBot = true; continue; }
                    botPairs.push_back({vb(i) - bottomShift, Vb.col(i)});
                    }
                if (!nanBot)
                    break;
                }
            }
        catch (const std::exception& e)
            {
            std::cerr << "Note: Lanczos solve of the most negative modes for " << name
                      << " failed internally (" << e.what() << "); retrying with a larger subspace.\n";
            }
        nconvBot = 0;
        botPairs.clear();
        }
    const bool bottomResolved = !botPairs.empty();
    if (topOK && !timedOut && !bottomResolved)
        std::cerr << "Warning: lambda_min for " << name << " did not converge"
                  << (bottomTimedOut ? " within its time cap" : "") << "; lambda_2 and the worst-case IACT are "
                     "reported, but lambda_min, the SLEM, the spectral gap, the relaxation time, and the "
                     "mixing-time bound are left unresolved (NaN).\n";

    // Assemble the spectrum: the known unit eigenvalue, the slow eigenvalues (lambda_2, ...), and the
    // most-negative eigenvalues, with residuals measured against S directly. Returns true if valid.
    auto assemble = [&](SpectralInfo& info) -> bool
        {
        info = SpectralInfo();
        info.computed_sparse = true;
        info.n_converged = nconvTop + nconvBot;
        info.ncv_used = ncvUsed;
        info.max_iterations_used = maxitUsed;
        info.tolerance_used = tolUsed;
        info.num_solver_attempts = attemptNumber;
        info.spectral_status = "insufficient_converged_eigenvalues";

        std::vector<std::pair<double, Eigen::VectorXd>> pairs;
        pairs.push_back({1.0, w});
        for (auto& pr : topPairs)
            pairs.push_back(pr);
        for (auto& pr : botPairs)
            pairs.push_back(pr);
        std::sort(pairs.begin(), pairs.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        std::vector<std::pair<double, Eigen::VectorXd>> uniq;
        for (auto& pr : pairs)
            {
            if (!uniq.empty() && std::abs(pr.first - uniq.back().first) < 1e-12)
                continue;
            uniq.push_back(pr);
            }

        Eigen::Index m = static_cast<Eigen::Index>(uniq.size());
        info.eigenvalues_real.resize(m);
        for (Eigen::Index i = 0; i < m; ++i)
            info.eigenvalues_real(i) = uniq[static_cast<size_t>(i)].first;
        finalizeSpectralReal(info);

        // Residual ||S u - lambda u||: lambda_1 uses the deterministic w-residual; the rest via S.
        auto residualOf = [&](const std::pair<double, Eigen::VectorXd>& pr) -> double
            {
            const Eigen::VectorXd& u = pr.second;
            double denom = std::max(u.norm(), std::numeric_limits<double>::min());
            Eigen::VectorXd Su(u.size());
            op.perform_op(u.data(), Su.data());
            return (Su - pr.first * u).norm() / denom;
            };
        double topResidual = lambda1Residual;
        double lambda2Top  = 0.0;
        for (auto& pr : topPairs)
            {
            topResidual = std::max(topResidual, residualOf(pr));
            lambda2Top  = std::max(lambda2Top, std::abs(pr.first));
            }
        // The most negative eigenpair matters only through the SLEM. Where the bottom of the spectrum is
        // a dense cluster near zero its Ritz vector converges slowly, although the value is long settled;
        // there it is accepted whenever |lambda_min|, widened by its residual, cannot be the SLEM.
        double bottomResidual = 0.0;
        bool   bottomOK       = true;
        for (auto& pr : botPairs)
            {
            double r = residualOf(pr);
            bottomResidual = std::max(bottomResidual, r);
            if (!(r < 1e-5) && !(std::abs(pr.first) + r < lambda2Top))
                bottomOK = false;
            }
        info.max_eigen_residual = std::max(topResidual, bottomResidual);
        info.lambda1_error = lambda1Residual;          // deterministic, from the known eigenpair

        bool enough     = (m >= 2) && !topPairs.empty();
        bool lambda1OK  = std::isfinite(info.lambda1_error) && info.lambda1_error < 1e-5;
        bool residualOK = std::isfinite(topResidual) && topResidual < 1e-5 && bottomOK;
        double lambda2Check = bottomResolved ? info.lambda2_abs : info.lambda2_algebraic;
        bool lambda2OK  = std::isfinite(lambda2Check) && lambda2Check <= 1.0 + 1e-7;

        // Without lambda_min the SLEM is unknown: finalizeSpectralReal took it from the slow modes alone,
        // which could understate it, so everything built on it is withdrawn. The worst-case IACT depends
        // on lambda_2 only and stands.
        if (!bottomResolved)
            {
            info.lambda_min      = std::numeric_limits<double>::quiet_NaN();
            info.lambda2_abs     = std::numeric_limits<double>::quiet_NaN();
            info.spectral_gap    = std::numeric_limits<double>::quiet_NaN();
            info.relaxation_time = std::numeric_limits<double>::quiet_NaN();
            }

        info.spectral_valid = enough && lambda1OK && residualOK && lambda2OK;
        if (info.spectral_valid)
            info.spectral_status = partialTop ? "ok_partial" : (bottomResolved ? "ok" : "ok_lambda_min_unresolved");
        else if (!enough)
            info.spectral_status = "insufficient_converged_eigenvalues";
        else if (!lambda1OK)
            info.spectral_status = "dominant_eigenvalue_not_close_to_one";
        else if (!residualOK)
            info.spectral_status = "large_eigen_residual";
        else if (!lambda2OK)
            info.spectral_status = "lambda2_modulus_greater_than_one";
        else
            info.spectral_status = "invalid_unknown_reason";
        return info.spectral_valid;
        };

    SpectralInfo info;
    bool valid = topOK && !timedOut && assemble(info);

    // A plain Lanczos result that converged by its own test but fails the residual check against S
    // (a near-degenerate cluster can do this) gets the shift-invert solve as well.
    if (!valid && topOK && !timedOut && !usedShiftInvert)
        {
        partialTop = false;
        if (runShiftInvert() && !timedOut)
            valid = assemble(info);
        }

    if (valid)
        {
        // Keep the slow eigenpairs for the Kemeny estimate, which deflates them exactly.
        std::sort(topPairs.begin(), topPairs.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        slowModeValues.resize(static_cast<Eigen::Index>(topPairs.size()));
        slowModeVectors.resize(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(topPairs.size()));
        for (size_t i = 0; i < topPairs.size(); ++i)
            {
            slowModeValues(static_cast<Eigen::Index>(i)) = topPairs[i].first;
            slowModeVectors.col(static_cast<Eigen::Index>(i)) = topPairs[i].second;
            }
        return info;
        }

    if (!topOK || timedOut)
        {
        info = SpectralInfo();
        info.computed_sparse = true;
        info.num_solver_attempts = attemptNumber;
        info.spectral_status = "insufficient_converged_eigenvalues";
        }

    if (timedOut)
        {
        info.spectral_status = "solver_timed_out";
        info.spectral_valid  = false;
        info.spectral_gap    = std::numeric_limits<double>::quiet_NaN();
        info.relaxation_time = std::numeric_limits<double>::quiet_NaN();
        info.worst_case_iact = std::numeric_limits<double>::quiet_NaN();
        std::cerr << "Note: eigensolver for " << name << " exceeded the " << s_solveBudgetSeconds
                  << " s budget; reporting the spectral gap as unresolved (solver_timed_out).\n";
        return info;
        }

    std::cerr << "Warning: sparse eigen calculation unreliable for " << name
              << " (status=" << info.spectral_status
              << ", converged=" << info.n_converged
              << ", max residual=" << info.max_eigen_residual
              << ", lambda1 error=" << info.lambda1_error
              << ").\n";
    info.spectral_gap = quietNaNValue();
    info.lambda2_abs = quietNaNValue();
    info.relaxation_time = quietNaNValue();
    info.worst_case_iact = quietNaNValue();
    return info;
}

// The k slowest modes of S (lambda_2, ..., lambda_{k+1}), by Lanczos on A^{-1}, A = I - S + w w', each
// application of A^{-1} being a Jacobi-preconditioned conjugate-gradient solve. A has eigenvalue
// 1 - lambda on every eigenvector of S orthogonal to w, and eigenvalue 1 on w itself, so the largest
// eigenvalues mu of A^{-1} are the slow modes, lambda = 1 - 1/mu. The eigenvectors are shared. If the
// spectrum below 1 is entirely negative, w (mu = 1) can appear among the k; the caller screens it by its
// overlap with w. Returns true if all k converged; otherwise lambdas and U hold whichever did.
bool MarkovChainAnalyzer::shiftInvertTopModes(int k, const Vector& wUnit, int maxRestarts, std::chrono::steady_clock::time_point deadline,
                                              bool& timedOut, Eigen::VectorXd& lambdas, DenseMatrix& U) const {

    using SolveClock = std::chrono::steady_clock;
    Eigen::Index N = static_cast<Eigen::Index>(n);
    k = std::max(1, std::min(k, static_cast<int>(n) - 2));
    int ncv = std::min(static_cast<int>(n), 2 * k + 8);

    if (isSparse)
        ensureRowSparse();
    Vector diagP = isSparse ? Vector(P_row_sparse.diagonal()) : Vector(P_dense.diagonal());
    Vector diagonal(N);
    for (Eigen::Index i = 0; i < N; ++i)
        diagonal(i) = std::max(1.0e-12, 1.0 - diagP(i) + wUnit(i) * wUnit(i));

    // A y = (I - S) y + w (w' y), formed in out without a temporary, its vector work threaded on large
    // kernels (as in conjugateGradient)
    std::vector<double> partial(64);
    ThreadPool* pool = (N >= (Eigen::Index(1) << 20)) ? threadPool : nullptr;
    auto applyA = [&](const Vector& y, Vector& out)
        {
        out.resize(N);
        applyS(y.data(), out.data());                             // S y
        const double* yp = y.data();
        const double* wp = wUnit.data();
        double*       op = out.data();
        forRowRanges(pool, N, 64, [&](Eigen::Index lo, Eigen::Index hi, int part)
            {
            double acc[4] = { 0.0, 0.0, 0.0, 0.0 };
            for (Eigen::Index i = lo; i < hi; ++i)
                acc[i & 3] += wp[i] * yp[i];
            partial[static_cast<size_t>(part)] = (acc[0] + acc[1]) + (acc[2] + acc[3]);
            });
        double wy = 0.0;
        for (double v : partial)
            wy += v;
        forRowRanges(pool, N, 64, [&](Eigen::Index lo, Eigen::Index hi, int)
            {
            for (Eigen::Index i = lo; i < hi; ++i)
                op[i] = (yp[i] - op[i]) + wp[i] * wy;
            });
        };

    long   totalCgIterations = 0;
    int    numSolves         = 0;
    int    numUnconverged    = 0;
    double worstResidual     = 0.0;
    const double cgTolerance = 1e-10;
    std::function<void(const double*, double*)> solve = [&](const double* x, double* y)
        {
        if (SolveClock::now() >= deadline)
            {
            timedOut = true;
            throw std::runtime_error("solve budget spent");
            }
        Vector b = Eigen::Map<const Vector>(x, N);
        Vector sol;
        int    iterations = 0;
        double residual   = 0.0;
        if (!conjugateGradient(applyA, diagonal, b, sol, cgTolerance, 100000, iterations, residual))
            ++numUnconverged;
        totalCgIterations += iterations;
        ++numSolves;
        worstResidual = std::max(worstResidual, residual);
        Eigen::Map<Vector>(y, N) = sol;
        };

    ShiftInvertOp siop(&solve, N);
    auto t0 = SolveClock::now();
    bool ok = false;
    try
        {
        Spectra::SymEigsSolver<ShiftInvertOp> eigs(siop, k, ncv);
        eigs.init();
        eigs.compute(Spectra::SortRule::LargestAlge, maxRestarts, 1e-9);
        ok = (eigs.info() == Spectra::CompInfo::Successful);
        Eigen::VectorXd mu = eigs.eigenvalues();                  // the converged Ritz values only
        lambdas.resize(mu.size());
        for (Eigen::Index i = 0; i < mu.size(); ++i)
            lambdas(i) = 1.0 - 1.0 / mu(i);
        U = eigs.eigenvectors();
        }
    catch (const std::exception& e)
        {
        if (!timedOut)
            std::cerr << "Note: shift-invert solve for " << name << " failed internally (" << e.what() << ").\n";
        ok = false;
        }

    double secs = std::chrono::duration<double>(SolveClock::now() - t0).count();
    std::cerr << "   [" << name << " top, shift-invert] " << numSolves << " CG solves, " << totalCgIterations
              << " CG iterations, " << secs << " s, converged=" << ok
              << (timedOut ? " (budget spent)" : "") << "\n";
    if (numUnconverged > 0)
        std::cerr << "Warning: " << numUnconverged << " of the shift-invert CG solves for " << name
                  << " stopped short of tolerance (worst relative residual " << worstResidual << ").\n";
    return ok && !timedOut;
}

ThresholdedIrreducibilityInfo MarkovChainAnalyzer::computeThresholdedIrreducibilityInfo(const std::vector<double>& thresholds, double tinyLeaveProb) const {

    ThresholdedIrreducibilityInfo summary;
    summary.thresholds = thresholds;
    summary.results.reserve(thresholds.size());

    if (isSparse && n > 0)
        {
        // One threaded pass gathers the degree summaries for every threshold. The transitions above a
        // threshold include those above any larger one, so once every state is reachable at some
        // threshold it is at every smaller one, and the searches there are skipped.
        std::vector<IrreducibilityInfo> infos(thresholds.size());
        for (size_t t = 0; t < thresholds.size(); ++t)
            {
            infos[t].threshold = thresholds[t];
            infos[t].num_states = n;
            infos[t].min_out_degree = std::numeric_limits<size_t>::max();
            infos[t].max_out_degree = 0;
            infos[t].min_leave_probability = std::numeric_limits<double>::infinity();
            }
        outDegreeStats(thresholds, tinyLeaveProb, infos);

        std::vector<size_t> order(thresholds.size());
        for (size_t t = 0; t < order.size(); ++t)
            order[t] = t;
        std::stable_sort(order.begin(), order.end(),
                         [&](size_t a, size_t b) { return thresholds[a] > thresholds[b]; });
        double fullForwardAt = -1.0, fullReverseAt = -1.0;       // a threshold with every state reached
        for (size_t t : order)
            {
            IrreducibilityInfo& info = infos[t];
            if (info.min_out_degree == std::numeric_limits<size_t>::max())
                info.min_out_degree = 0;
            if (!std::isfinite(info.min_leave_probability))
                info.min_leave_probability = std::numeric_limits<double>::quiet_NaN();
            double tau = thresholds[t];
            info.states_reachable_from_0 = (fullForwardAt >= tau) ? n : countReachableForward(reachabilityReference, tau);
            info.states_that_can_reach_0 = (fullReverseAt >= tau) ? n : countReachableReverse(reachabilityReference, tau);
            if (info.states_reachable_from_0 == n && fullForwardAt < tau)
                fullForwardAt = tau;
            if (info.states_that_can_reach_0 == n && fullReverseAt < tau)
                fullReverseAt = tau;
            info.irreducible = (info.states_reachable_from_0 == n && info.states_that_can_reach_0 == n);
            }
        for (size_t t = 0; t < thresholds.size(); ++t)
            {
            summary.results.push_back(infos[t]);
            if (thresholds[t] == 0.0)
                summary.irreducible_at_zero = infos[t].irreducible;
            if (infos[t].irreducible)
                summary.largest_threshold_irreducible = thresholds[t];
            }
        return summary;
        }

    for (double tau : thresholds)
        {
        IrreducibilityInfo info = computeIrreducibilityInfo(tau, tinyLeaveProb);
        summary.results.push_back(info);
        if (tau == 0.0)
            summary.irreducible_at_zero = info.irreducible;
        if (info.irreducible)
            summary.largest_threshold_irreducible = tau;
        }
    return summary;
}

TransitionProbabilityInfo MarkovChainAnalyzer::computeTransitionProbabilityInfo(const std::vector<double>& thresholds) const {

    TransitionProbabilityInfo info;
    info.thresholds = thresholds;
    info.num_transitions_le_threshold.assign(thresholds.size(), 0);
    info.num_states_leave_le_threshold.assign(thresholds.size(), 0);

    Eigen::Index N = static_cast<Eigen::Index>(n);
    double sumTrans = 0.0;
    double sumLeave = 0.0;
    info.min_positive_offdiag_transition = std::numeric_limits<double>::infinity();
    info.max_offdiag_transition = 0.0;
    info.min_leave_probability = std::numeric_limits<double>::infinity();
    info.max_leave_probability = 0.0;

    auto recordTransition = [&](double p) {
        if (p <= 0.0)
            return;
        ++info.num_positive_offdiag_transitions;
        sumTrans += p;
        info.min_positive_offdiag_transition = std::min(info.min_positive_offdiag_transition, p);
        info.max_offdiag_transition = std::max(info.max_offdiag_transition, p);
        for (size_t k = 0; k < thresholds.size(); ++k)
            if (p <= thresholds[k])
                ++info.num_transitions_le_threshold[k];
    };

    auto recordLeaveProbability = [&](double leaveProb) {
        sumLeave += leaveProb;
        info.min_leave_probability = std::min(info.min_leave_probability, leaveProb);
        info.max_leave_probability = std::max(info.max_leave_probability, leaveProb);
        if (leaveProb <= 0.0)
            ++info.num_states_with_zero_leave_probability;
        for (size_t k = 0; k < thresholds.size(); ++k)
            if (leaveProb <= thresholds[k])
                ++info.num_states_leave_le_threshold[k];
    };

    if (isSparse)
        {
        ensureRowSparse();
        for (Eigen::Index i = 0; i < N; ++i)
            {
            double leaveProb = 0.0;
            for (RowSparseMatrix::InnerIterator it(P_row_sparse, i); it; ++it)
                {
                Eigen::Index j = it.col();
                double p = it.value();
                if (j == i)
                    continue;
                leaveProb += p;
                recordTransition(p);
                }
            recordLeaveProbability(leaveProb);
            }
        }
    else
        {
        for (Eigen::Index i = 0; i < N; ++i)
            {
            double leaveProb = 0.0;
            for (Eigen::Index j = 0; j < N; ++j)
                {
                if (j == i)
                    continue;
                double p = P_dense(i, j);
                leaveProb += p;
                recordTransition(p);
                }
            recordLeaveProbability(leaveProb);
            }
        }

    if (info.num_positive_offdiag_transitions > 0)
        {
        info.mean_positive_offdiag_transition = sumTrans / static_cast<double>(info.num_positive_offdiag_transitions);
        }
    else
        {
        info.min_positive_offdiag_transition = std::numeric_limits<double>::quiet_NaN();
        info.max_offdiag_transition = std::numeric_limits<double>::quiet_NaN();
        info.mean_positive_offdiag_transition = std::numeric_limits<double>::quiet_NaN();
        }

    if (n > 0)
        info.mean_leave_probability = sumLeave / static_cast<double>(n);
    else
        {
        info.min_leave_probability = std::numeric_limits<double>::quiet_NaN();
        info.max_leave_probability = std::numeric_limits<double>::quiet_NaN();
        info.mean_leave_probability = std::numeric_limits<double>::quiet_NaN();
        }

    if (!std::isfinite(info.min_leave_probability))
        info.min_leave_probability = std::numeric_limits<double>::quiet_NaN();

    return info;
}

double MarkovChainAnalyzer::conductanceFromOrderingFast(const std::vector<Eigen::Index>& order) const {

    if (order.size() < 2)
        return std::numeric_limits<double>::quiet_NaN();

    Eigen::Index N = static_cast<Eigen::Index>(n);
    std::vector<char> inSet(static_cast<size_t>(N), 0);
    double mass = 0.0;
    double flow = 0.0; // Q(A, A^c)
    double best = std::numeric_limits<double>::infinity();

    if (isSparse) 
        {
        ensureRowSparse();
        for (size_t k = 0; k + 1 < order.size(); ++k) 
            {
            Eigen::Index x = order[k];
            if (inSet[static_cast<size_t>(x)]) continue;

            // Remove old A -> x contributions, because x moves from A^c to A. Each is pi_i P(i,x), which
            // by detailed balance is pi_x P(x,i), so they are read from row x.
            for (RowSparseMatrix::InnerIterator it(P_row_sparse, x); it; ++it)
                {
                Eigen::Index i = it.col();
                if (inSet[static_cast<size_t>(i)])
                    flow -= pi(x) * it.value();
                }

            // Add new x -> A^c contributions.
            for (RowSparseMatrix::InnerIterator it(P_row_sparse, x); it; ++it) 
                {
                Eigen::Index j = it.col();
                if (!inSet[static_cast<size_t>(j)] && j != x)
                    flow += pi(x) * it.value();
                }

            inSet[static_cast<size_t>(x)] = 1;
            mass += pi(x);
            double denom = std::min(mass, 1.0 - mass);
            if (denom > 0.0)
                best = std::min(best, flow / denom);
            if (mass > 0.5)
                break;
            }
        } 
    else 
        {
        for (size_t k = 0; k + 1 < order.size(); ++k) 
            {
            Eigen::Index x = order[k];
            if (inSet[static_cast<size_t>(x)]) continue;
            for (Eigen::Index i = 0; i < N; ++i)
                if (inSet[static_cast<size_t>(i)])
                    flow -= pi(i) * P_dense(i, x);
            for (Eigen::Index j = 0; j < N; ++j)
                if (!inSet[static_cast<size_t>(j)] && j != x)
                    flow += pi(x) * P_dense(x, j);
            inSet[static_cast<size_t>(x)] = 1;
            mass += pi(x);
            double denom = std::min(mass, 1.0 - mass);
            if (denom > 0.0)
                best = std::min(best, flow / denom);
            if (mass > 0.5)
                break;
            }
        }

    if (!std::isfinite(best))
        return std::numeric_limits<double>::quiet_NaN();
    return std::max(0.0, best);
}

std::vector<double> MarkovChainAnalyzer::defaultIrreducibilityThresholds(void) {

    return {0.0, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6};
}

MarkovChainAnalyzer::DenseMatrix MarkovChainAnalyzer::denseTransitionMatrix(void) const {

    return isSparse ? DenseMatrix(P_row_sparse) : P_dense;
}

// The efficiency, or variance ratio, of the chain for estimating the posterior mean of h.
//
// The argument is an indicator: one for every topology carrying the feature of interest -- a given
// topology, or every topology containing a given split -- and zero for all the rest. Its posterior
// mean, E_pi[h] = p, is then that feature's posterior probability, and h is a Bernoulli variable, so
// that independent sampling from the posterior would estimate p with variance p(1-p)/t after t
// draws. The chain does worse, because its successive states are correlated: the variance of its
// estimate is var_pi(h, P)/t, in which
//
//     var_pi(h, P) = var_pi(h) + 2 sum_{t >= 1} Cov( h(tau_s), h(tau_{s+t}) )
//
// is the asymptotic variance. The ratio of the two variances is the efficiency: t iterations of the
// chain carry as much information about p as t * efficiency independent draws from the posterior.
//
// The asymptotic variance is most naturally understood through the eigenvalues, but it need not be
// computed from them, which matters because the full spectrum is unavailable at these sizes. It
// follows instead from the solution of Poisson's equation, (I - P) ghat = g, in which
// g = h - E_pi[h] is the centered indicator. That operator is singular, but adding the stationary
// matrix Pi repairs it without changing the solution, because Pi g = 0 for any centered g.
// Symmetrizing by the substitution ghat = D^{-1/2} y, with D = diag(pi), turns the system into
//
//     (I - S + s s') y = D^{1/2} g,      S = D^{1/2} P D^{-1/2},   s = sqrt(pi),
//
// which is symmetric and positive definite because the chain is reversible. Writing c = D^{1/2} g,
//
//     var_pi(h)    = c . c            (what independent sampling would give)
//     var_pi(h, P) = 2 c . y - c . c  (what the chain gives)
//
// so a single conjugate-gradient solve, costing one sparse product with the kernel per iteration,
// yields the efficiency exactly for any function of interest.
EfficiencyInfo MarkovChainAnalyzer::efficiencyFor(const Vector& h, int maxIterations, double tolerance) const {

    EfficiencyInfo info;

    Eigen::Index N = static_cast<Eigen::Index>(n);
    if (h.size() != N)
        return info;
    if (isSparse)
        ensureRowSparse();

    Vector piSqrt = posteriorSqrt();

    double p = pi.dot(h);
    info.posterior_mean = p;

    Vector g = h - Vector::Constant(N, p);
    Vector c = piSqrt.cwiseProduct(g);                            // c = D^{1/2} g

    double varIndependent = c.dot(c);                             // p(1-p) when h is an indicator
    info.independent_variance = varIndependent;
    if (!(varIndependent > 0.0))
        return info;                                              // h is constant under the posterior

    Vector diagP = isSparse ? Vector(P_row_sparse.diagonal()) : Vector(P_dense.diagonal());
    Vector diagonal(N);
    for (Eigen::Index i = 0; i < N; ++i)
        diagonal(i) = std::max(1.0e-12, 1.0 - diagP(i) + pi(i));

    auto applyA = [&](const Vector& y, Vector& out)
        {
        applySymmetrizedLaplacian(piSqrt, y, out);
        out += piSqrt * piSqrt.dot(y);                            // the rank-one repair, s s' y
        };

    Vector y;
    info.converged = conjugateGradient(applyA, diagonal, c, y, tolerance, maxIterations,
                                       info.iterations, info.relative_residual);
    if (info.converged == false)
        std::cerr << "Warning: the efficiency calculation for " << name << " stopped after "
                  << info.iterations << " iterations with a relative residual of "
                  << info.relative_residual << ".\n";

    double varChain = 2.0 * c.dot(y) - varIndependent;
    info.asymptotic_variance = varChain;
    if (varChain > 0.0)
        {
        info.efficiency = varIndependent / varChain;
        info.integrated_autocorrelation_time = varChain / varIndependent;
        }
    return info;
}

double MarkovChainAnalyzer::eigenvectorSweepConductance(size_t maxDenseStates) const {

    if (n > maxDenseStates)
        return std::numeric_limits<double>::quiet_NaN();

    // The kernel is reversible, so the right eigenvectors of P are v = D^{-1/2} u, u the eigenvectors of
    // the symmetrized S, and the cached symmetric eigensystem serves here. (Eigen's general, complex
    // EigenSolver on P, used before, was some fifty times slower than the symmetric solve.)
    if (!ensureDenseEigensystem())
        return std::numeric_limits<double>::quiet_NaN();

    std::vector<std::pair<double, Eigen::Index>> moduli;
    moduli.reserve(static_cast<size_t>(n));
    for (Eigen::Index i = 0; i < denseEigenvalues.size(); ++i)
        moduli.push_back({std::abs(denseEigenvalues(i)), i});
    std::stable_sort(moduli.begin(), moduli.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    if (moduli.size() < 2)
        return std::numeric_limits<double>::quiet_NaN();

    Eigen::Index eigIndex = moduli[1].second;
    Vector v = pi.cwiseSqrt().cwiseInverse().cwiseProduct(denseEigenvectors.col(eigIndex));
    std::vector<Eigen::Index> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&v](Eigen::Index a, Eigen::Index b) {
        return v(a) < v(b);
    });

    // An eigenvector's sign is arbitrary, and the sweep stops at half the posterior mass, so a sweep in
    // one direction sees only the cuts on one side of the median. Sweeping both ways covers every cut
    // of the ordering and makes the estimate independent of the sign the solver happened to return.
    double forward = conductanceFromOrderingFast(order);
    std::reverse(order.begin(), order.end());
    double backward = conductanceFromOrderingFast(order);
    if (std::isnan(forward))
        return backward;
    if (std::isnan(backward))
        return forward;
    return std::min(forward, backward);
}

void MarkovChainAnalyzer::ensureRowSparse(void) const {

    // The kernel is built in row-major form, so there is nothing to convert. Kept so that the routines
    // that once triggered the conversion need not change.
}

double MarkovChainAnalyzer::entropyRate(void) const {

    double H = 0.0;
    Eigen::Index N = static_cast<Eigen::Index>(n);
    if (isSparse) 
        {
        ensureRowSparse();
        for (Eigen::Index i = 0; i < N; i++) 
            {
            for (RowSparseMatrix::InnerIterator it(P_row_sparse, i); it; ++it) 
                {
                double p = it.value();
                if (p > 0.0)
                    H -= pi(i) * p * std::log(p);
                }
            }
        } 
    else 
        {
        for (Eigen::Index i = 0; i < N; i++)
            for (Eigen::Index j = 0; j < N; j++)
                if (P_dense(i, j) > 0.0)
                    H -= pi(i) * P_dense(i, j) * std::log(P_dense(i, j));
        }
    return H;
}

void MarkovChainAnalyzer::finalizeSpectralReal(SpectralInfo& info) {

    const Eigen::VectorXd& ev = info.eigenvalues_real; // sorted in descending algebraic order
    Eigen::Index m = ev.size();
    if (m == 0)
        return;

    info.eigenvalue_moduli.resize(m);
    info.eigenvalues_complex.resize(m);
    info.multiplicity_of_1 = 0;
    info.has_complex_eigenvalues = false;

    double slem   = 0.0;
    double lam2   = -std::numeric_limits<double>::infinity(); // largest eigenvalue strictly below 1
    double lammin =  std::numeric_limits<double>::infinity();

    for (Eigen::Index i = 0; i < m; ++i)
        {
        double lam = ev(i);
        info.eigenvalue_moduli(i) = std::abs(lam);
        info.eigenvalues_complex(i) = std::complex<double>(lam, 0.0);
        lammin = std::min(lammin, lam);
        if (std::abs(lam - 1.0) < 1e-8)
            {
            ++info.multiplicity_of_1;
            }
        else
            {
            slem = std::max(slem, std::abs(lam));
            lam2 = std::max(lam2, lam);
            }
        }

    info.lambda_min        = std::isfinite(lammin) ? lammin : std::numeric_limits<double>::quiet_NaN();
    info.lambda2_algebraic = std::isfinite(lam2)   ? lam2   : std::numeric_limits<double>::quiet_NaN();
    info.lambda2_abs       = slem;
    info.spectral_gap      = 1.0 - slem;

                            // Guard: a spectral radius above 1 (an eigenvalue > 1, i.e. a
                            // non-reversible or noisy kernel) or a repeated unit eigenvalue
                            // (reducible) makes the gap, relaxation time, and any 1/(1-lambda)
                            // quantity meaningless. Report NaN rather than a negative gap /
                            // infinite relaxation / large-negative Kemeny. lambda2_abs,
                            // lambda2_algebraic, and lambda_min are kept so the cause is visible.
    bool unreliableSpectrum = (info.multiplicity_of_1 > 1) || (slem > 1.0 + 1e-6);
    if (unreliableSpectrum)
        {
        info.spectral_gap    = std::numeric_limits<double>::quiet_NaN();
        info.relaxation_time = std::numeric_limits<double>::quiet_NaN();
        info.worst_case_iact = std::numeric_limits<double>::quiet_NaN();
        if (info.spectral_status == "not_computed" || info.spectral_status.empty())
            info.spectral_status = (info.multiplicity_of_1 > 1) ? "reducible" : "spectral_radius_gt_1";
        return;
        }

    if (info.spectral_gap > 0.0)
        {
        info.relaxation_time = 1.0 / info.spectral_gap;
                            // Worst-case IACT is governed by the slowest *positive* mode (largest
                            // algebraic eigenvalue below 1), not the SLEM: a near -1 eigenvalue is
                            // anti-correlated and has IACT < 1, so it never dominates the variance.
        double l = info.lambda2_algebraic;
        info.worst_case_iact = (std::isfinite(l) && l < 1.0)
                             ? (1.0 + l) / (1.0 - l)
                             : std::numeric_limits<double>::infinity();
        }
    else
        {
        info.relaxation_time = std::numeric_limits<double>::infinity();
        info.worst_case_iact = std::numeric_limits<double>::infinity();
        }
}

void MarkovChainAnalyzer::finalizeSpectralInfo(SpectralInfo& info) {

    if (info.eigenvalue_moduli.size() > 1) 
        {
        info.lambda2_abs = info.eigenvalue_moduli(1);
        info.spectral_gap = 1.0 - info.lambda2_abs;
        if (info.spectral_gap > 0.0) 
            {
            info.relaxation_time = 1.0 / info.spectral_gap;
            info.worst_case_iact = (1.0 + info.lambda2_abs) / info.spectral_gap;
            } 
        else 
            {
            info.relaxation_time = std::numeric_limits<double>::infinity();
            info.worst_case_iact = std::numeric_limits<double>::infinity();
            }
        }
}

bool MarkovChainAnalyzer::fullSymmetricEigensystem(Eigen::VectorXd& evals, DenseMatrix& U) const {

    if (n < 1 || !ensureDenseEigensystem())
        return false;
    evals = denseEigenvalues.reverse();             // ascending
    U = denseEigenvectors.rowwise().reverse();      // columns are Euclidean-orthonormal eigenvectors of S
    return true;
}

// The full eigensystem of S = D^{1/2} P D^{-1/2}, computed once and cached: eigenvalues in descending
// order and the matching orthonormal eigenvectors. Every dense spectral routine (the spectral summary,
// Kemeny's constant, the eigenvector sweep, the eigensystem files) reads it, where each used to run its
// own O(n^3) solve -- four per (move, power) cell, three of them repeats.
bool MarkovChainAnalyzer::ensureDenseEigensystem(void) const {

    if (denseEigenReady)
        return true;
    if (n < 1)
        return false;

    DenseMatrix S = symmetrizedDense();
    Eigen::Index N = S.rows();
    Eigen::VectorXd ascending(N);

#if MCA_USE_LAPACK
    {
    int nn = static_cast<int>(N), lda = nn, info = 0, lwork = -1, liwork = -1, iworkQuery = 0;
    double workQuery = 0.0;
    const char jobz = 'V', uplo = 'L';
    dsyevd_(&jobz, &uplo, &nn, S.data(), &lda, ascending.data(), &workQuery, &lwork, &iworkQuery, &liwork, &info);
    if (info == 0)
        {
        lwork  = static_cast<int>(workQuery);
        liwork = iworkQuery;
        std::vector<double> work(static_cast<size_t>(std::max(1, lwork)));
        std::vector<int>    iwork(static_cast<size_t>(std::max(1, liwork)));
        dsyevd_(&jobz, &uplo, &nn, S.data(), &lda, ascending.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
        }
    if (info != 0)
        {
        std::cerr << "Note: LAPACK dsyevd failed for " << name << " (info = " << info
                  << "); falling back to Eigen's self-adjoint solver.\n";
        S = symmetrizedDense();
        Eigen::SelfAdjointEigenSolver<DenseMatrix> es(S);
        if (es.info() != Eigen::Success)
            return false;
        ascending = es.eigenvalues();
        S = es.eigenvectors();
        }
    }
#else
    {
    Eigen::SelfAdjointEigenSolver<DenseMatrix> es(S);
    if (es.info() != Eigen::Success)
        return false;
    ascending = es.eigenvalues();
    S = es.eigenvectors();
    }
#endif

    // Reorder to descending, swapping columns in place rather than copying an n x n matrix.
    denseEigenvalues = ascending.reverse();
    for (Eigen::Index k = 0; k < N / 2; ++k)
        S.col(k).swap(S.col(N - 1 - k));
    denseEigenvectors = std::move(S);
    denseEigenReady = true;
    return true;
}

SpectralInfo MarkovChainAnalyzer::getSpectralInfo(int nev) const {

    nev = std::max(2, nev);
    if (spectralCacheValid && spectralCacheNev >= nev)
        return spectralCache;
    spectralCache = (isSparse && n > denseStateLimit) ? computeSpectralInfoSparse(nev) : computeSpectralInfo();
    spectralCacheNev = nev;
    spectralCacheValid = true;
    return spectralCache;
}

Eigen::Index MarkovChainAnalyzer::getMAPTreeIndex(void) const {

    Eigen::Index idx = 0;
    pi.maxCoeff(&idx);
    return idx;
}

MarkovChainAnalyzer::Vector MarkovChainAnalyzer::hittingTimesToStateVector(Eigen::Index target) const {

    if (target < 0 || target >= static_cast<Eigen::Index>(n))
        return Vector();

    Eigen::Index N = static_cast<Eigen::Index>(n);
    Vector b = Vector::Ones(N);
    b(target) = 0.0;

    DenseMatrix P = denseTransitionMatrix();
    DenseMatrix A = DenseMatrix::Identity(N, N) - P;
    A.row(target).setZero();
    A(target, target) = 1.0;

    Eigen::PartialPivLU<DenseMatrix> solver(A);
    return solver.solve(b);
}

MarkovChainAnalyzer::Vector MarkovChainAnalyzer::hittingTimesToSetVector(const std::vector<Eigen::Index>& targets) const {

    if (targets.empty())
        return Vector();

    Eigen::Index N = static_cast<Eigen::Index>(n);
    std::vector<char> isTarget(static_cast<size_t>(N), 0);
    for (Eigen::Index idx : targets)
        if (idx >= 0 && idx < N)
            isTarget[static_cast<size_t>(idx)] = 1;

    Vector b = Vector::Ones(N);
    for (Eigen::Index i = 0; i < N; ++i)
        if (isTarget[static_cast<size_t>(i)])
            b(i) = 0.0;

    DenseMatrix P = denseTransitionMatrix();
    DenseMatrix A = DenseMatrix::Identity(N, N) - P;
    for (Eigen::Index i = 0; i < N; ++i)
        {
        if (isTarget[static_cast<size_t>(i)])
            {
            A.row(i).setZero();
            A(i, i) = 1.0;
            }
        }

    Eigen::PartialPivLU<DenseMatrix> solver(A);
    return solver.solve(b);
}

// The indicator of a single topology: one at that tree and zero everywhere else. Its posterior mean
// is the tree's own posterior probability.
MarkovChainAnalyzer::Vector MarkovChainAnalyzer::indicatorForTree(uint64_t treeHash) const {

    Vector h = Vector::Zero(static_cast<Eigen::Index>(n));
    Eigen::Index idx = stateIndexForHash(treeHash);
    if (idx < 0)
        {
        if (!supportWasRestricted)
            std::cerr << "Warning: tree " << treeHash << " is not among the states of " << name << ".\n";
        }
    else
        h(idx) = 1.0;
    return h;
}

// The indicator of a set of topologies: one at every tree in the set and zero everywhere else. Its
// posterior mean is the posterior probability of the set, so passing the trees that carry one of the
// MAP tree's splits gives an indicator whose posterior mean is that split's posterior probability.
MarkovChainAnalyzer::Vector MarkovChainAnalyzer::indicatorForTrees(const std::vector<uint64_t>& treeHashes) const {

    Vector h = Vector::Zero(static_cast<Eigen::Index>(n));
    size_t numMissing = 0;
    for (uint64_t treeHash : treeHashes)
        {
        Eigen::Index idx = stateIndexForHash(treeHash);
        if (idx < 0)
            numMissing++;
        else
            h(idx) = 1.0;
        }
    if (numMissing > 0 && !supportWasRestricted)
        std::cerr << "Warning: " << numMissing << " of " << treeHashes.size()
                  << " trees are not among the states of " << name << ".\n";
    return h;
}

MarkovChainAnalyzer::Vector MarkovChainAnalyzer::indicatorMAP(void) const {

    Vector f = Vector::Zero(static_cast<Eigen::Index>(n));
    Eigen::Index idx = getMAPTreeIndex();
    if (idx >= 0 && idx < static_cast<Eigen::Index>(n))
        f(idx) = 1.0;
    return f;
}

MarkovChainAnalyzer::Vector MarkovChainAnalyzer::indicatorPosteriorMassSet(double targetMass) const {

    Vector f = Vector::Zero(static_cast<Eigen::Index>(n));
    for (Eigen::Index idx : posteriorMassSet(targetMass))
        if (idx >= 0 && idx < static_cast<Eigen::Index>(n))
            f(idx) = 1.0;
    return f;
}

void MarkovChainAnalyzer::initCommon(void) {

    double s = pi.sum();
    if (!(s > 0.0))
        throw std::runtime_error("Posterior has non-positive sum (" + std::to_string(s) + ")");
    if (std::abs(s - 1.0) > 1e-8)
        std::cerr << "Note: normalizing posterior (sum was " << s << ")\n";
    pi /= s;
}

double MarkovChainAnalyzer::integratedAutocorrelationTime(const Vector& f) const {

    // tau_int(f) = sum_{j>=2} a_j^2 (1+lambda_j)/(1-lambda_j) / Var_pi(f),
    // with a_j = < D^{1/2} f, u_j >, the projection of f onto the pi-orthonormal eigenbasis.
    //
    // For n <= denseStateLimit the full spectrum is available and this is exact. For larger n only
    // the leading slow modes (and a couple of the most-negative modes) are computed; the remaining
    // ~n bulk modes have lambda ~ 0 and each contribute ~1 to the autocorrelation sum. Normalizing
    // by the *captured* variance instead of the true variance -- as an earlier version did -- let
    // the ratio collapse onto (1+lambda_2)/(1-lambda_2) whenever one slow mode dominated the
    // captured subspace, reporting the worst-case IACT for every functional regardless of its true
    // overlap with the slow mode. We instead normalize by the exact stationary variance and credit
    // the uncaptured mass with unit IACT, so the truncated estimate degrades gracefully rather than
    // diverging. When an exact value is needed at any n, use efficiencyFor(), which returns the same
    // quantity from a single sparse solve of Poisson's equation.
    if (f.size() != static_cast<Eigen::Index>(n))
        return std::numeric_limits<double>::quiet_NaN();

    Eigen::VectorXd evals;
    DenseMatrix U;
    if (!spectralModesForFunctionals(evals, U))
        return std::numeric_limits<double>::quiet_NaN();

    Vector g = pi.cwiseSqrt().cwiseProduct(f);                   // D^{1/2} f
    Eigen::VectorXd a = U.transpose() * g;

    double totalVariance = stationaryVariance(f);                // sum_{j>=2} a_j^2, exact
    if (!(totalVariance > 0.0))
        return std::numeric_limits<double>::quiet_NaN();

    double num         = 0.0;                                    // sum over captured modes j >= 2
    double capturedVar = 0.0;
    for (Eigen::Index j = 0; j < evals.size(); ++j)
        {
        double lam = evals(j);
        if (std::abs(lam - 1.0) < 1e-8)                          // skip the stationary mode
            continue;
        double a2 = a(j) * a(j);
        capturedVar += a2;
        num         += a2 * (1.0 + lam) / (1.0 - lam);
        }

    double uncapturedVar = std::max(0.0, totalVariance - capturedVar);
    num += uncapturedVar;                                        // bulk modes: lambda ~ 0 => IACT ~ 1
    return num / totalVariance;
}

double MarkovChainAnalyzer::kemenyConstant(void) const {

    if (n > denseStateLimit) 
        {
        std::cout << "   * Exact Kemeny's constant skipped for n = " << n
                  << "; use kemenyConstantStochastic() for a stochastic-trace estimate\n";
        return std::numeric_limits<double>::quiet_NaN();
        }

    SpectralInfo spec = computeSpectralInfo();   // dense self-adjoint => full real spectrum

    // Kemeny's constant is a sum of 1/(1-lambda_j) with every lambda_j < 1, so it is strictly
    // positive and finite for an irreducible reversible chain. A repeated unit eigenvalue
    // (reducible) or an eigenvalue >= 1 (non-reversible / noisy) makes the sum meaningless.
    if (spec.multiplicity_of_1 > 1 || !std::isfinite(spec.lambda2_abs) || spec.lambda2_abs > 1.0 + 1e-6)
        return std::numeric_limits<double>::quiet_NaN();

    double K = 0.0;
    for (Eigen::Index i = 0; i < spec.eigenvalues_real.size(); i++) 
        {
        double lambda = spec.eigenvalues_real(i);
        if (std::abs(lambda - 1.0) < 1e-8)
            continue;
        K += 1.0 / (1.0 - lambda);
        }
    return (K >= 0.0) ? K : std::numeric_limits<double>::quiet_NaN();
}

double MarkovChainAnalyzer::kemenyConstantStochastic(int numProbes, int lanczosSteps, unsigned int seed) const {

    // K = sum_{j>=2} 1/(1 - lambda_j) = trace( (I - S)^+ ), the trace of the pseudo-inverse of
    // M = I - S over the complement of its null space. S = D^{1/2} P D^{-1/2} is symmetric with
    // the same spectrum as P, and w = sqrt(pi) is its exact stationary eigenvector (M w = 0).
    //
    // The trace is split in two. The slow modes found by the spectral solve (lambda_2, lambda_3, ...)
    // are summed exactly. On a chain whose basins are weakly connected they dominate the sum -- a
    // single lambda_2 = 1 - 6.5e-8 contributes 1.5e7 -- and a Hutchinson estimate, which samples the
    // trace through random probes, cannot resolve a sum carried by a handful of directions: an earlier
    // version, which estimated the whole trace stochastically, came out more than ten times too small on
    // such a chain. The remainder, the trace over the complement of w and those modes, is estimated by
    // stochastic Lanczos quadrature: for Rademacher probes z projected onto the complement, Hutchinson
    // gives E[ z'^T f(M) z' ] = trace of f(M) there, f(mu) = 1/mu, and each quadratic form is the Gauss
    // quadrature read off an m-step Lanczos tridiagonalization of M.
    //
    // The probes are run as blocks, so that each pass over the kernel serves every probe in the block;
    // the product is bandwidth-bound, and a block of eight costs little more than a single probe. Each
    // Lanczos step is reorthogonalized against the deflated directions and its two predecessors only.
    // Full reorthogonalization would hold m vectors per probe -- gigabytes per block at ten taxa -- and
    // Gauss quadrature from Lanczos without it remains accurate (lost orthogonality duplicates converged
    // Ritz values, but their quadrature weights split among the copies).
    if (!isSparse)
        {
        // dense fallback: exact when affordable
        if (n <= denseStateLimit)
            return kemenyConstant();
        return std::numeric_limits<double>::quiet_NaN();
        }
    if (n < 2)
        return std::numeric_limits<double>::quiet_NaN();

    Eigen::Index N = static_cast<Eigen::Index>(n);

    Vector w = pi.cwiseSqrt();
    double wnorm = w.norm();
    if (!(wnorm > 0.0))
        return std::numeric_limits<double>::quiet_NaN();
    w /= wnorm;

    // ---- The exact part: the slow modes from the spectral solve ----
    std::vector<double> slowLambdas;
    std::vector<Vector> deflation;                                // orthonormal: w, then the slow modes
    deflation.push_back(w);
    if (n > denseStateLimit)
        {
        SpectralInfo spec = getSpectralInfo(defaultSparseEigenvalues);
        if (spec.spectral_valid && spec.multiplicity_of_1 > 1)
            return std::numeric_limits<double>::quiet_NaN();    // reducible: K is not defined
        if (spec.spectral_valid)
            {
            for (Eigen::Index j = 0; j < slowModeValues.size(); ++j)
                {
                double lam = slowModeValues(j);
                if (!(lam < 1.0 - 1e-8))
                    continue;
                Vector u = slowModeVectors.col(j);
                for (int pass = 0; pass < 2; ++pass)              // Gram-Schmidt, twice for safety
                    for (const Vector& q : deflation)
                        u -= q.dot(u) * q;
                double un = u.norm();
                if (un < 0.5)                                     // not independent of those kept
                    continue;
                deflation.push_back(u / un);
                slowLambdas.push_back(lam);
                }
            }
        else
            {
            std::cerr << "Warning: the slow modes of " << name << " are unresolved (" << spec.spectral_status
                      << "); the Kemeny estimate is purely stochastic and can badly underestimate a chain "
                         "with a small spectral gap.\n";
            }
        }

    double Kslow = 0.0;
    for (double lam : slowLambdas)
        Kslow += 1.0 / (1.0 - lam);

    DenseMatrix Q(N, static_cast<Eigen::Index>(deflation.size()));
    for (size_t j = 0; j < deflation.size(); ++j)
        Q.col(static_cast<Eigen::Index>(j)) = deflation[j];
    std::vector<Vector>().swap(deflation);

    using RowBlock = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    auto deflate = [&](RowBlock& X)
        {
        DenseMatrix C = Q.transpose() * X;
        X.noalias() -= Q * C;
        };

    // ---- The stochastic part: block SLQ on the complement ----
    int m = std::max(2, std::min(lanczosSteps, static_cast<int>(N) - 1));
    const int blockSize = 8;
    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<int> coin(0, 1);

    double traceEstimate = 0.0;
    int validProbes = 0;

    for (int p0 = 0; p0 < numProbes; p0 += blockSize)
        {
        int nb = std::min(blockSize, numProbes - p0);

        // the signs are drawn in the trees' map order, so the probes do not depend on the state numbering
        RowBlock Z(N, nb);
        for (int c = 0; c < nb; ++c)
            for (Eigen::Index k = 0; k < N; ++k)
                Z(mapOrderState.empty() ? k : mapOrderState[static_cast<size_t>(k)], c) = coin(gen) ? 1.0 : -1.0;
        deflate(Z);

        std::vector<double> z2(static_cast<size_t>(nb));
        std::vector<char>   active(static_cast<size_t>(nb), 0);
        RowBlock Qcur = RowBlock::Zero(N, nb), Qprev = RowBlock::Zero(N, nb), SQ(N, nb), R(N, nb);
        for (int c = 0; c < nb; ++c)
            {
            z2[static_cast<size_t>(c)] = Z.col(c).squaredNorm();
            if (z2[static_cast<size_t>(c)] > 0.0)
                {
                Qcur.col(c) = Z.col(c) / std::sqrt(z2[static_cast<size_t>(c)]);
                active[static_cast<size_t>(c)] = 1;
                }
            }
        Z.resize(0, 0);

        std::vector<std::vector<double>> alpha(static_cast<size_t>(nb)), beta(static_cast<size_t>(nb));
        std::vector<double> betaPrev(static_cast<size_t>(nb), 0.0);

        // Each Lanczos step is a handful of sweeps over the rows, each fusing the column operations that
        // can share it, run in parallel over a fixed partition of the rows with the partial sums combined
        // in a fixed order. Done column by column with Eigen, as earlier, the step made some hundred
        // strided passes over the n x 8 blocks on one core -- a third of a second per step at ten taxa,
        // 200 steps per estimate, more than the products with the kernel themselves on an NNI kernel.
        const int    parts = 64;
        const int    d     = static_cast<int>(Q.cols());
        const double* qp   = Q.data();                            // column-major n x d
        const size_t  nbs  = static_cast<size_t>(nb);
        std::vector<double> partial(static_cast<size_t>(parts) * static_cast<size_t>(std::max(d, 1)) * nbs);
        std::vector<double> coef(static_cast<size_t>(std::max(d, 1)) * nbs);
        std::vector<double> colA(nbs), colB(nbs);

        // coef = Q^T X (d x nb), from per-part sums
        auto projectOnQ = [&](const double* X)
            {
            std::fill(partial.begin(), partial.end(), 0.0);
            forRowRanges(threadPool, N, parts, [&](Eigen::Index lo, Eigen::Index hi, int part)
                {
                double* acc = partial.data() + static_cast<size_t>(part) * static_cast<size_t>(d) * nbs;
                for (int j = 0; j < d; ++j)
                    {
                    const double* qj = qp + static_cast<size_t>(j) * static_cast<size_t>(N);
                    double* aj = acc + static_cast<size_t>(j) * nbs;
                    for (Eigen::Index i = lo; i < hi; ++i)
                        {
                        double q = qj[i];
                        const double* xi = X + i * nb;
                        for (int c = 0; c < nb; ++c)
                            aj[c] += q * xi[c];
                        }
                    }
                });
            std::fill(coef.begin(), coef.end(), 0.0);
            for (int part = 0; part < parts; ++part)
                for (size_t t = 0; t < static_cast<size_t>(d) * nbs; ++t)
                    coef[t] += partial[static_cast<size_t>(part) * static_cast<size_t>(d) * nbs + t];
            };

        // X -= Q coef, then out[c] = sum_i f(X_ic, Y_ic) over the updated rows (Y may be null)
        auto subtractQ = [&](double* X, const double* Y, bool squares, std::vector<double>& out)
            {
            std::fill(partial.begin(), partial.end(), 0.0);
            forRowRanges(threadPool, N, parts, [&](Eigen::Index lo, Eigen::Index hi, int part)
                {
                double* acc = partial.data() + static_cast<size_t>(part) * nbs;
                for (Eigen::Index i = lo; i < hi; ++i)
                    {
                    double* xi = X + i * nb;
                    for (int j = 0; j < d; ++j)
                        {
                        double q = qp[static_cast<size_t>(j) * static_cast<size_t>(N) + i];
                        const double* cj = coef.data() + static_cast<size_t>(j) * nbs;
                        for (int c = 0; c < nb; ++c)
                            xi[c] -= q * cj[c];
                        }
                    const double* yi = (Y != nullptr) ? Y + i * nb : xi;
                    for (int c = 0; c < nb; ++c)
                        acc[c] += squares ? xi[c] * xi[c] : yi[c] * xi[c];
                    }
                });
            std::fill(out.begin(), out.end(), 0.0);
            for (int part = 0; part < parts; ++part)
                for (size_t c = 0; c < nbs; ++c)
                    out[c] += partial[static_cast<size_t>(part) * nbs + c];
            };

        // R[:,c] -= s[c] * V[:,c] for active c, then out[c] = sum_i W_ic R_ic (W null: R_ic^2)
        auto axpyThenDot = [&](double* Rp, const double* V, const std::vector<double>& sc,
                               const double* W, std::vector<double>& out)
            {
            std::fill(partial.begin(), partial.end(), 0.0);
            forRowRanges(threadPool, N, parts, [&](Eigen::Index lo, Eigen::Index hi, int part)
                {
                double* acc = partial.data() + static_cast<size_t>(part) * nbs;
                for (Eigen::Index i = lo; i < hi; ++i)
                    {
                    double* ri = Rp + i * nb;
                    const double* vi = V + i * nb;
                    const double* wi = (W != nullptr) ? W + i * nb : ri;
                    for (int c = 0; c < nb; ++c)
                        {
                        if (active[static_cast<size_t>(c)])
                            ri[c] -= sc[static_cast<size_t>(c)] * vi[c];
                        acc[c] += wi[c] * ri[c];
                        }
                    }
                });
            std::fill(out.begin(), out.end(), 0.0);
            for (int part = 0; part < parts; ++part)
                for (size_t c = 0; c < nbs; ++c)
                    out[c] += partial[static_cast<size_t>(part) * nbs + c];
            };

        double* qc = Qcur.data();
        double* qv = Qprev.data();
        double* sq = SQ.data();
        double* rp = R.data();

        for (int k = 0; k < m; ++k)
            {
            // deflate the current vectors and normalize them; retired columns are zeroed
            projectOnQ(qc);
            subtractQ(qc, nullptr, true, colA);                   // colA = squared column norms
            bool anyActive = false;
            for (int c = 0; c < nb; ++c)
                {
                size_t cc = static_cast<size_t>(c);
                colB[cc] = 0.0;                                   // 0: zero the column
                if (!active[cc])
                    continue;
                double qn = std::sqrt(colA[cc]);
                if (qn <= 1e-14)
                    {
                    active[cc] = 0;
                    continue;
                    }
                colB[cc] = qn;
                anyActive = true;
                }
            if (!anyActive)
                break;
            forRowRanges(threadPool, N, parts, [&](Eigen::Index lo, Eigen::Index hi, int)
                {
                for (Eigen::Index i = lo; i < hi; ++i)
                    for (int c = 0; c < nb; ++c)
                        qc[i * nb + c] = (colB[static_cast<size_t>(c)] > 0.0) ? qc[i * nb + c] / colB[static_cast<size_t>(c)] : 0.0;
                });

            applySBlock(qc, sq, nb);

            // R = (I - S) q, alpha = q . R
            std::fill(partial.begin(), partial.end(), 0.0);
            forRowRanges(threadPool, N, parts, [&](Eigen::Index lo, Eigen::Index hi, int part)
                {
                double* acc = partial.data() + static_cast<size_t>(part) * nbs;
                for (Eigen::Index i = lo; i < hi; ++i)
                    for (int c = 0; c < nb; ++c)
                        {
                        double r = qc[i * nb + c] - sq[i * nb + c];
                        rp[i * nb + c] = r;
                        acc[c] += qc[i * nb + c] * r;
                        }
                });
            std::fill(colA.begin(), colA.end(), 0.0);
            for (int part = 0; part < parts; ++part)
                for (size_t c = 0; c < nbs; ++c)
                    colA[c] += partial[static_cast<size_t>(part) * nbs + c];
            for (int c = 0; c < nb; ++c)
                if (active[static_cast<size_t>(c)])
                    alpha[static_cast<size_t>(c)].push_back(colA[static_cast<size_t>(c)]);

            // R -= alpha q + beta_prev q_prev, and project R on the deflated directions
            forRowRanges(threadPool, N, parts, [&](Eigen::Index lo, Eigen::Index hi, int)
                {
                for (Eigen::Index i = lo; i < hi; ++i)
                    for (int c = 0; c < nb; ++c)
                        if (active[static_cast<size_t>(c)])
                            rp[i * nb + c] -= colA[static_cast<size_t>(c)] * qc[i * nb + c] +
                                              betaPrev[static_cast<size_t>(c)] * qv[i * nb + c];
                });
            projectOnQ(rp);
            subtractQ(rp, qc, false, colA);                       // deflate R; colA = q . R

            // local reorthogonalization against q and q_prev, then beta = |R|
            axpyThenDot(rp, qc, colA, qv, colB);                  // R -= (q.R) q;  colB = q_prev . R
            std::vector<double> bsq(nbs);
            axpyThenDot(rp, qv, colB, nullptr, bsq);              // R -= (q_prev.R) q_prev;  bsq = |R|^2

            for (int c = 0; c < nb; ++c)
                {
                size_t cc = static_cast<size_t>(c);
                colB[cc] = 0.0;                                   // 0: leave the column as it is
                if (!active[cc])
                    continue;
                double b = std::sqrt(bsq[cc]);
                if (b <= 1e-14 || k == m - 1)
                    {
                    active[cc] = 0;
                    continue;
                    }
                beta[cc].push_back(b);
                betaPrev[cc] = b;
                colB[cc] = b;
                }
            forRowRanges(threadPool, N, parts, [&](Eigen::Index lo, Eigen::Index hi, int)
                {
                for (Eigen::Index i = lo; i < hi; ++i)
                    for (int c = 0; c < nb; ++c)
                        {
                        double b = colB[static_cast<size_t>(c)];
                        if (b > 0.0)
                            {
                            qv[i * nb + c] = qc[i * nb + c];
                            qc[i * nb + c] = rp[i * nb + c] / b;
                            }
                        }
                });
            }

        for (int c = 0; c < nb; ++c)
            {
            size_t cc = static_cast<size_t>(c);
            int steps = static_cast<int>(alpha[cc].size());
            if (steps < 1)
                continue;

            Eigen::MatrixXd T = Eigen::MatrixXd::Zero(steps, steps);
            for (int k = 0; k < steps; ++k)
                {
                T(k, k) = alpha[cc][static_cast<size_t>(k)];
                if (k + 1 < steps)
                    {
                    T(k, k + 1) = beta[cc][static_cast<size_t>(k)];
                    T(k + 1, k) = beta[cc][static_cast<size_t>(k)];
                    }
                }

            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
            if (es.info() != Eigen::Success)
                continue;
            Eigen::VectorXd theta = es.eigenvalues();
            Eigen::MatrixXd Y = es.eigenvectors();

            double quad = 0.0;
            for (int k = 0; k < steps; ++k)
                {
                double th = theta(k);
                double tau0 = Y(0, k);
                if (th > 1e-12)                                   // skip the (deflated) null direction
                    quad += tau0 * tau0 / th;
                }

            traceEstimate += z2[cc] * quad;
            ++validProbes;
            }
        }

    if (validProbes == 0)
        return std::numeric_limits<double>::quiet_NaN();
    double K = Kslow + traceEstimate / static_cast<double>(validProbes);
    // Kemeny's constant is strictly positive for an irreducible reversible chain; a negative
    // estimate means M = I - S has an eigenvalue <= 0 (spectral radius of P above 1), i.e. the
    // kernel is not pi-reversible. Report NaN rather than a spurious large-negative value.
    if (!(K >= 0.0))
        {
        std::cerr << "Warning: stochastic Kemeny estimate is negative for " << name
                  << " (kernel not pi-reversible); returning NaN.\n";
        return std::numeric_limits<double>::quiet_NaN();
        }
    return K;
}

bool MarkovChainAnalyzer::leadingSymmetricEigensystem(int kLargest, int kSmallest, Eigen::VectorXd& evals, DenseMatrix& U) const {

    if (!isSparse || n < 2)
        return false;

    using Op = AnalyzerSymOp;
    Op op(this, static_cast<Eigen::Index>(n));

    auto solveEnd = [&](int k, Spectra::SortRule rule,
                        std::vector<double>& outVals, std::vector<Eigen::VectorXd>& outVecs) -> bool {
        k = std::max(1, std::min(k, static_cast<int>(n) - 1));
        int ncv = std::min(static_cast<int>(n), std::max(2 * k + 1, 20));
        ncv = std::max(ncv, k + 2);
        ncv = std::min(ncv, static_cast<int>(n));
        Spectra::SymEigsSolver<Op> eigs(op, k, ncv);
        eigs.init();
        {
        using LClock = std::chrono::steady_clock;
        const LClock::time_point lDeadline =
            LClock::now() + std::chrono::duration_cast<LClock::duration>(
                                std::chrono::duration<double>(s_solveBudgetSeconds));
        for (int done = 0; done < 6000; done += 200)
            {
            eigs.compute(rule, std::min(200, 6000 - done), 1e-12);
            if (eigs.info() == Spectra::CompInfo::Successful) break;
            if (LClock::now() >= lDeadline) break;
            }
        }
        if (eigs.info() != Spectra::CompInfo::Successful)
            return false;
        Eigen::VectorXd v = eigs.eigenvalues();
        Eigen::MatrixXd V = eigs.eigenvectors();
        for (Eigen::Index i = 0; i < v.size(); ++i)
            {
            outVals.push_back(v(i));
            outVecs.push_back(V.col(i));
            }
        return true;
    };

    std::vector<double> vals;
    std::vector<Eigen::VectorXd> vecs;

    // Largest-algebraic gives lambda_1 = 1 and the slowest positive modes (lambda_2, ...),
    // which are exactly the modes that dominate relaxation and functional IACT.
    if (!solveEnd(kLargest, Spectra::SortRule::LargestAlge, vals, vecs))
        return false;
    // Smallest-algebraic captures the most negative eigenvalue (near-periodicity).
    if (kSmallest > 0)
        solveEnd(kSmallest, Spectra::SortRule::SmallestAlge, vals, vecs);

    if (vals.size() < 2)
        return false;

    // Deduplicate (the two solves can overlap) and sort descending by algebraic value.
    std::vector<size_t> order(vals.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return vals[a] > vals[b]; });

    std::vector<size_t> kept;
    for (size_t idx : order)
        {
        if (!kept.empty() && std::abs(vals[idx] - vals[kept.back()]) < 1e-12)
            continue;
        kept.push_back(idx);
        }

    evals.resize(static_cast<Eigen::Index>(kept.size()));
    U.resize(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(kept.size()));
    for (size_t c = 0; c < kept.size(); ++c)
        {
        evals(static_cast<Eigen::Index>(c)) = vals[kept[c]];
        U.col(static_cast<Eigen::Index>(c)) = vecs[kept[c]];
        }
    return true;
}

// Mean first-passage times to a single target topology, from every starting topology at once.
// Conditioning on the first move of the chain gives
//
//     m_i = 1 + sum_{k != j} P_ik m_kj      (i != j),      m_jj = 0,
//
// which says that the chain takes one iteration, at a cost of one, after which it has either
// arrived at the target or finds itself at some other topology from which the journey must still
// be completed. Because m_jj = 0 the excluded term contributes nothing, and the system may be
// written as (I - P) m = 1 on every row but that of the target. Substituting m = D^{-1/2} y
// symmetrizes it to (I - S) y = sqrt(pi), to be solved on the subspace in which the target
// component of y vanishes. On that subspace the operator is positive definite -- its only null
// direction, sqrt(pi), has a non-zero target component and so does not lie there -- and conjugate
// gradient applies. One sparse solve therefore returns the expected passage time to the target
// from every one of the B(N) topologies. The matrix of all pairwise passage times has B(N)^2
// entries and could never be formed; the passage times to any one tree can be.
MarkovChainAnalyzer::Vector MarkovChainAnalyzer::meanFirstPassageTimesToState(Eigen::Index target, int maxIterations, double tolerance) const {

    Eigen::Index N = static_cast<Eigen::Index>(n);
    if (target < 0 || target >= N)
        return Vector();
    if (isSparse)
        ensureRowSparse();

    Vector piSqrt = posteriorSqrt();

    Vector b  = piSqrt;
    b(target) = 0.0;

    Vector diagP = isSparse ? Vector(P_row_sparse.diagonal()) : Vector(P_dense.diagonal());
    Vector diagonal(N);
    for (Eigen::Index i = 0; i < N; ++i)
        diagonal(i) = std::max(1.0e-12, 1.0 - diagP(i));
    diagonal(target) = 1.0;

    auto applyA = [&](const Vector& y, Vector& out)
        {
        applySymmetrizedLaplacian(piSqrt, y, out);
        out(target) = 0.0;                                        // project onto the subspace y_j = 0
        };

    Vector y;
    int    iterations       = 0;
    double relativeResidual = 0.0;
    if (conjugateGradient(applyA, diagonal, b, y, tolerance, maxIterations, iterations, relativeResidual) == false)
        std::cerr << "Warning: the mean first-passage times for " << name << " stopped after "
                  << iterations << " iterations with a relative residual of " << relativeResidual << ".\n";

    Vector m  = y.array() / piSqrt.array();
    m(target) = 0.0;                                              // the target is reached in no time at all
    return m;
}

// The mean first-passage times to the tree with a given hash. The MapTree class identifies the MAP
// tree and its partitions by hash, so this is the natural entry point when the target comes from
// there rather than from the posterior probabilities held here.
MarkovChainAnalyzer::Vector MarkovChainAnalyzer::meanFirstPassageTimesToTree(uint64_t treeHash, int maxIterations, double tolerance) const {

    Eigen::Index target = stateIndexForHash(treeHash);
    if (target < 0)
        {
        if (!supportWasRestricted)
            std::cerr << "Warning: tree " << treeHash << " is not among the states of " << name << ".\n";
        return Vector();
        }
    return meanFirstPassageTimesToState(target, maxIterations, tolerance);
}

double MarkovChainAnalyzer::meanHittingTimeToMAP(void) const {

    return meanHittingTimeToState(getMAPTreeIndex());
}

double MarkovChainAnalyzer::meanHittingTimeToState(Eigen::Index target) const {

    // The average is taken with equal weight over every topology, so this is the expected number of
    // iterations needed to first reach the target from a tree drawn from the prior, which is uniform
    // over topologies. That is the quantity of interest, because an MCMC analysis begins at a tree
    // drawn from the prior. The mean first-passage times are obtained by conjugate gradient, so no
    // limit on the size of the state space is needed.
    if (target < 0 || target >= static_cast<Eigen::Index>(n))
        return std::numeric_limits<double>::quiet_NaN();

    Vector m = meanFirstPassageTimesToState(target);
    if (m.size() == 0)
        return std::numeric_limits<double>::quiet_NaN();

    double avg = m.mean();
    std::cout << "   * MAP/target tree index: " << target << " (posterior = " << pi(target) << ")\n";
    std::cout << "   * Average hitting time to target (uniform start): " << avg << " steps\n";
    return avg;
}

double MarkovChainAnalyzer::meanHittingTimeToSet(const std::vector<Eigen::Index>& targets) const {

    if (targets.empty())
        return std::numeric_limits<double>::quiet_NaN();
    if (n > exactHittingStateLimit) 
        {
        std::cout << "   * Hitting time to set: skipped for n = " << n
                  << " because exact sparse linear solve is too expensive at this size\n";
        return std::numeric_limits<double>::quiet_NaN();
        }

    Eigen::Index N = static_cast<Eigen::Index>(n);
    std::vector<char> isTarget(static_cast<size_t>(N), 0);
    for (Eigen::Index idx : targets)
        if (idx >= 0 && idx < N)
            isTarget[static_cast<size_t>(idx)] = 1;

    Vector b = Vector::Ones(N);
    for (Eigen::Index i = 0; i < N; i++)
        if (isTarget[static_cast<size_t>(i)])
            b(i) = 0.0;

    Vector h;
    if (isSparse) 
        {
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(static_cast<size_t>(P_row_sparse.nonZeros() + N));
        ensureRowSparse();
        for (Eigen::Index i = 0; i < N; i++) 
            {
            if (isTarget[static_cast<size_t>(i)]) 
                {
                triplets.emplace_back(i, i, 1.0);
                } 
            else 
                {
                triplets.emplace_back(i, i, 1.0);
                for (RowSparseMatrix::InnerIterator it(P_row_sparse, i); it; ++it)
                    triplets.emplace_back(i, it.col(), -it.value());
                }
            }
        SparseMatrix A(N, N);
        A.setFromTriplets(triplets.begin(), triplets.end());
        A.makeCompressed();
        Eigen::SparseLU<SparseMatrix> solver;
        solver.analyzePattern(A);
        solver.factorize(A);
        if (solver.info() != Eigen::Success)
            return std::numeric_limits<double>::quiet_NaN();
        h = solver.solve(b);
        } 
    else 
        {
        DenseMatrix A = DenseMatrix::Identity(N, N) - P_dense;
        for (Eigen::Index i = 0; i < N; i++) 
            {
            if (isTarget[static_cast<size_t>(i)]) 
                {
                A.row(i) = DenseMatrix::Zero(1, N);
                A(i, i) = 1.0;
                }
            }
        Eigen::ColPivHouseholderQR<DenseMatrix> solver(A);
        h = solver.solve(b);
        }
    return h.mean();
}

double MarkovChainAnalyzer::meanHittingTimeToPosteriorMass(double targetMass) const {

    std::vector<Eigen::Index> targets = posteriorMassSet(targetMass);
    double mass = 0.0;
    for (Eigen::Index idx : targets)
        mass += pi(idx);

    std::cout << "   * Number of states in top posterior-mass set: " << targets.size() << "\n";
    std::cout << "   * Actual posterior mass of set: " << mass << "\n";

    double hit = meanHittingTimeToSet(targets);
    if (!std::isnan(hit))
        std::cout << "   * Average hitting time to top posterior-mass set (uniform start): " << hit << " steps\n";
    return hit;
}

double MarkovChainAnalyzer::meanHittingTimeToStateQuiet(Eigen::Index target) const {

    if (target < 0 || target >= static_cast<Eigen::Index>(n))
        return std::numeric_limits<double>::quiet_NaN();
    if (n > exactHittingStateLimit)
        return std::numeric_limits<double>::quiet_NaN();

    Eigen::Index N = static_cast<Eigen::Index>(n);
    Vector b = Vector::Ones(N);
    b(target) = 0.0;
    Vector h;

    if (isSparse) 
        {
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(static_cast<size_t>(P_row_sparse.nonZeros() + N));
        ensureRowSparse();
        for (Eigen::Index i = 0; i < N; ++i) 
            {
            if (i == target) 
                {
                triplets.emplace_back(i, i, 1.0);
                } 
            else 
                {
                triplets.emplace_back(i, i, 1.0);
                for (RowSparseMatrix::InnerIterator it(P_row_sparse, i); it; ++it)
                    triplets.emplace_back(i, it.col(), -it.value());
                }
            }
        SparseMatrix A(N, N);
        A.setFromTriplets(triplets.begin(), triplets.end());
        A.makeCompressed();
        Eigen::SparseLU<SparseMatrix> solver;
        solver.analyzePattern(A);
        solver.factorize(A);
        if (solver.info() != Eigen::Success)
            return std::numeric_limits<double>::quiet_NaN();
        h = solver.solve(b);
        if (solver.info() != Eigen::Success)
            return std::numeric_limits<double>::quiet_NaN();
        } 
    else 
        {
        DenseMatrix A = DenseMatrix::Identity(N, N) - P_dense;
        A.row(target) = DenseMatrix::Zero(1, N);
        A(target, target) = 1.0;
        Eigen::ColPivHouseholderQR<DenseMatrix> solver(A);
        h = solver.solve(b);
        }

    return h.mean();
}

double MarkovChainAnalyzer::meanHittingTimeToSetQuiet(const std::vector<Eigen::Index>& targets) const {

    if (targets.empty())
        return std::numeric_limits<double>::quiet_NaN();
    if (n > exactHittingStateLimit)
        return std::numeric_limits<double>::quiet_NaN();

    Eigen::Index N = static_cast<Eigen::Index>(n);
    std::vector<char> isTarget(static_cast<size_t>(N), 0);
    for (Eigen::Index idx : targets)
        if (idx >= 0 && idx < N)
            isTarget[static_cast<size_t>(idx)] = 1;

    Vector b = Vector::Ones(N);
    for (Eigen::Index i = 0; i < N; ++i)
        if (isTarget[static_cast<size_t>(i)])
            b(i) = 0.0;

    Vector h;
    if (isSparse) 
        {
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(static_cast<size_t>(P_row_sparse.nonZeros() + N));
        ensureRowSparse();
        for (Eigen::Index i = 0; i < N; ++i) 
            {
            if (isTarget[static_cast<size_t>(i)]) 
                {
                triplets.emplace_back(i, i, 1.0);
                } 
            else 
                {
                triplets.emplace_back(i, i, 1.0);
                for (RowSparseMatrix::InnerIterator it(P_row_sparse, i); it; ++it)
                    triplets.emplace_back(i, it.col(), -it.value());
                }
            }
        SparseMatrix A(N, N);
        A.setFromTriplets(triplets.begin(), triplets.end());
        A.makeCompressed();
        Eigen::SparseLU<SparseMatrix> solver;
        solver.analyzePattern(A);
        solver.factorize(A);
        if (solver.info() != Eigen::Success)
            return std::numeric_limits<double>::quiet_NaN();
        h = solver.solve(b);
        if (solver.info() != Eigen::Success)
            return std::numeric_limits<double>::quiet_NaN();
        } 
    else 
        {
        DenseMatrix A = DenseMatrix::Identity(N, N) - P_dense;
        for (Eigen::Index i = 0; i < N; ++i) 
            {
            if (isTarget[static_cast<size_t>(i)]) 
                {
                A.row(i) = DenseMatrix::Zero(1, N);
                A(i, i) = 1.0;
                }
            }
        Eigen::ColPivHouseholderQR<DenseMatrix> solver(A);
        h = solver.solve(b);
        }

    return h.mean();
}

MarkovChainAnalyzer::Vector MarkovChainAnalyzer::meanReturnTimes(void) const {

    return pi.array().inverse();
}

double MarkovChainAnalyzer::mixingTimeUpperBoundValue(double epsilon) const {

    SpectralInfo spec = getSpectralInfo(defaultSparseEigenvalues);
    if (!spec.spectral_valid)
        return std::numeric_limits<double>::quiet_NaN();
    double minPi = pi.minCoeff();
    if (minPi <= 0.0)
        minPi = 1.0 / static_cast<double>(n);
    if (!(spec.spectral_gap > 0.0))
        return std::numeric_limits<double>::quiet_NaN();
    return (1.0 / spec.spectral_gap) * std::log(1.0 / (epsilon * minPi));
}

double MarkovChainAnalyzer::mixingTimeUpperBound(double epsilon) const {

    SpectralInfo spec = getSpectralInfo(defaultSparseEigenvalues);
    if (!spec.spectral_valid || !(spec.spectral_gap > 0.0))
        {
        std::cout << "   * Mixing time upper bound: unavailable (spectral estimate invalid)\n";
        return std::numeric_limits<double>::quiet_NaN();
        }
    double minPi = pi.minCoeff();
    if (minPi <= 0.0)
        minPi = 1.0 / static_cast<double>(n);
    double bound = (1.0 / spec.spectral_gap) * std::log(1.0 / (epsilon * minPi));
    std::cout << "   * Mixing time upper bound (epsilon=" << epsilon << "): approx. " << bound << " steps\n";
    return bound;
}

PeskunComparison MarkovChainAnalyzer::peskunComparison(const MarkovChainAnalyzer& other, double tol) const {

    PeskunComparison r;

    std::unordered_map<uint64_t, Eigen::Index> idxThis;
    idxThis.reserve(stateHashes.size() * 2);
    for (size_t i = 0; i < stateHashes.size(); ++i)
        if (stateHashes[i])
            idxThis[stateHashes[i]] = static_cast<Eigen::Index>(i);

    for (size_t i = 0; i < other.stateHashes.size(); ++i)
        if (other.stateHashes[i] && idxThis.count(other.stateHashes[i]))
            ++r.shared_states;

    if (other.isSparse)
        other.ensureRowSparse();
    this->ensureRowSparse();

    Eigen::Index No = static_cast<Eigen::Index>(other.n);
    for (Eigen::Index oi = 0; oi < No; ++oi)
        {
        uint64_t hi = (static_cast<size_t>(oi) < other.stateHashes.size()) ? other.stateHashes[static_cast<size_t>(oi)] : 0;
        if (!hi)
            continue;
        auto itI = idxThis.find(hi);
        if (itI == idxThis.end())
            continue;
        Eigen::Index ti = itI->second;

        auto handle = [&](Eigen::Index oj, double pOther) {
            if (oj == oi || pOther <= 0.0)
                return;
            uint64_t hj = (static_cast<size_t>(oj) < other.stateHashes.size()) ? other.stateHashes[static_cast<size_t>(oj)] : 0;
            double pThis = 0.0;
            if (hj)
                {
                auto itJ = idxThis.find(hj);
                if (itJ != idxThis.end())
                    pThis = this->transitionProbability(ti, itJ->second);
                }
            ++r.compared_offdiag;
            double slack = pThis - pOther;
            r.min_slack = std::min(r.min_slack, slack);
            if (slack < -tol)
                {
                ++r.violations;
                r.max_violation = std::max(r.max_violation, -slack);
                }
        };

        if (other.isSparse)
            {
            for (RowSparseMatrix::InnerIterator it(other.P_row_sparse, oi); it; ++it)
                handle(it.col(), it.value());
            }
        else
            {
            for (Eigen::Index oj = 0; oj < No; ++oj)
                handle(oj, other.P_dense(oi, oj));
            }
        }

    r.dominates = (r.violations == 0 && r.compared_offdiag > 0);
    return r;
}

// The square root of the posterior probability of every topology, which the symmetrization
// S = D^{1/2} P D^{-1/2} requires. The inverse square root appears in that expression, so a topology
// whose posterior probability has underflowed to zero would produce an infinity and silently corrupt
// everything downstream. Such topologies are floored and reported. Underflow is a real possibility
// here: the posterior probabilities span the exponential of the range of log likelihoods, and for a
// data set with strong phylogenetic signal the worst trees can fall below the smallest representable
// double.
MarkovChainAnalyzer::Vector MarkovChainAnalyzer::posteriorSqrt(void) const {

    Eigen::Index N = static_cast<Eigen::Index>(n);
    Vector s(N);
    size_t numZero = 0;
    for (Eigen::Index i = 0; i < N; ++i)
        {
        if (pi(i) > 0.0)
            {
            s(i) = std::sqrt(pi(i));
            }
        else
            {
            s(i) = std::sqrt(std::numeric_limits<double>::min());
            numZero++;
            }
        }

    if (numZero > 0 && warnedZeroPosterior == false)
        {
        warnedZeroPosterior = true;
        std::cerr << "Warning: " << numZero << " of " << N << " topologies in " << name
                  << " have a posterior probability of zero, presumably through underflow. The "
                     "symmetrization of the kernel divides by the square root of the posterior "
                     "probability, so these are floored at the smallest representable value; "
                     "quantities that depend on them should be treated with caution.\n";
        }
    return s;
}

// The product of the transition kernel with a vector, formed in parallel. This is the innermost
// operation of every eigensolve and every linear solve in this class, and for the larger move sets
// it is where essentially all of the time is spent.
void MarkovChainAnalyzer::sparseMatVec(const Vector& x, Vector& y) const {

    ensureRowSparse();                                            // gather along rows, not scatter

    Eigen::Index N = static_cast<Eigen::Index>(n);
    y.resize(N);

    const int*    outerIndex = P_row_sparse.outerIndexPtr();
    const int*    innerIndex = P_row_sparse.innerIndexPtr();
    const double* values     = P_row_sparse.valuePtr();

    if (threadPool == nullptr)
        {
        y.noalias() = P_row_sparse * x;
        return;
        }

    int numTasks = std::max(1, threadPool->getThreadCount() * 4);  // oversubscribe: rows differ in length
    Eigen::Index chunk = (N + numTasks - 1) / numTasks;
    if (chunk < 1)
        chunk = 1;

    std::vector<SparseMatVecTask> tasks;
    tasks.reserve(static_cast<size_t>(numTasks) + 1);
    for (Eigen::Index lo = 0; lo < N; lo += chunk)
        tasks.emplace_back(outerIndex, innerIndex, values, x.data(), y.data(), lo, std::min(N, lo + chunk));
    for (SparseMatVecTask& t : tasks)
        threadPool->pushTask(&t);
    threadPool->wait();
}

std::vector<Eigen::Index> MarkovChainAnalyzer::posteriorMassSet(double targetMass) const {

    std::vector<Eigen::Index> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [this](Eigen::Index a, Eigen::Index b) {
        return pi(a) > pi(b);
    });

    std::vector<Eigen::Index> targets;
    double mass = 0.0;
    for (Eigen::Index idx : order) 
        {
        targets.push_back(idx);
        mass += pi(idx);
        if (mass >= targetMass)
            break;
        }
    return targets;
}

double MarkovChainAnalyzer::posteriorSweepConductance(void) const {

    std::vector<Eigen::Index> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [this](Eigen::Index a, Eigen::Index b) {
        return pi(a) > pi(b);
    });
    return conductanceFromOrderingFast(order);
}

void MarkovChainAnalyzer::printReport(std::ostream& os) const {

    os << "   Analysis for " << name << "\n";
    os << "   * Number of trees: " << n << "\n";

    verifyStationary();
    if (n <= detailedBalanceStateLimit)
        checkDetailedBalance();
    else
        os << "   * Detailed balance: skipped for large n\n";

    checkIrreducible(0.0, os);
    ThresholdedIrreducibilityInfo thrInfo = computeThresholdedIrreducibilityInfo();
    TransitionProbabilityInfo transInfo = computeTransitionProbabilityInfo();
    os << "   * Largest transition-probability threshold preserving irreducibility: "
       << thrInfo.largest_threshold_irreducible << "\n";
    os << "   * Smallest positive off-diagonal transition probability: "
       << transInfo.min_positive_offdiag_transition << "\n";
    os << "   * Mean probability of leaving a state: "
       << transInfo.mean_leave_probability << "\n";
    os << "   * Minimum probability of leaving a state: "
       << transInfo.min_leave_probability << "\n";

    SpectralInfo spec = getSpectralInfo(defaultSparseEigenvalues);
    os << "   * SLEM |lambda_2|: " << spec.lambda2_abs << "\n";
    os << "   * Spectral gap, 1 - SLEM: " << spec.spectral_gap << "\n";
    os << "   * Relaxation time approx.: " << spec.relaxation_time << "\n";
    os << "   * Worst-case IACT approx.: " << spec.worst_case_iact << "\n";
    os << "   * Sparse eigen calculation: " << (spec.computed_sparse ? "yes" : "no") << "\n";
    os << "   * Eigenvalues converged/computed: " << spec.n_converged << "\n";
    os << "   * Eigenvalue 1 multiplicity among computed eigenvalues: " << spec.multiplicity_of_1 << "\n";
    os << "   * Complex eigenvalues among computed eigenvalues: " << (spec.has_complex_eigenvalues ? "yes" : "no") << "\n";
    os << "   * Average acceptance rate: " << averageAcceptanceRate() << "\n";
    os << "   * Entropy rate: " << entropyRate() << " nats/step\n";

    if (n <= denseStateLimit)
        os << "   * Kemeny's constant: " << kemenyConstant() << "\n";
    else
        os << "   * Approximate Kemeny's constant from leading eigenvalues: " << approximateKemenyConstant(32) << "\n";

    os << "   * Posterior-ordered sweep conductance estimate: " << posteriorSweepConductance() << "\n";

    double eigCond = eigenvectorSweepConductance();
    if (std::isnan(eigCond))
        os << "   * Eigenvector sweep conductance estimate: skipped for large n\n";
    else
        os << "   * Eigenvector sweep conductance estimate: " << eigCond << "\n";

    Vector mrt = meanReturnTimes();
    os << "   * Mean return time range: [" << mrt.minCoeff() << ", " << mrt.maxCoeff() << "]\n";
}

void MarkovChainAnalyzer::printExtendedReport(std::ostream& os) const {

    printReport(os);
    if (n <= exactHittingStateLimit) 
        {
        meanHittingTimeToMAP();
        meanHittingTimeToPosteriorMass(0.95);
        } 
    else 
        {
        std::cout << "   * Exact hitting-time diagnostics: skipped for large n\n";
        std::vector<Eigen::Index> targets = posteriorMassSet(0.95);
        double mass = 0.0;
        for (Eigen::Index idx : targets) mass += pi(idx);
        std::cout << "   * Top 95% posterior-mass set size: " << targets.size()
                  << " (actual mass = " << mass << ")\n";
        }
    mixingTimeUpperBound(1e-6);
}

bool MarkovChainAnalyzer::spectralModesForFunctionals(Eigen::VectorXd& evals, DenseMatrix& U) const {

    // Returns eigenpairs of S in descending algebraic order. Exact (full spectrum) for small n;
    // otherwise the leading slow modes, which carry the dominant contribution to IACT / chi^2.
    if (n <= denseStateLimit)
        {
        if (!ensureDenseEigensystem())
            return false;
        evals = denseEigenvalues;
        U = denseEigenvectors;
        return true;
        }
    int kL = std::max(defaultSparseEigenvalues, 16);
    return leadingSymmetricEigensystem(kL, 2, evals, U);
}

// The index of the state holding a given tree, or -1 if the tree is not in the state space. The
// lookup is constant time: converting a set of trees, such as those carrying one of the MAP tree's
// splits, into an indicator vector would otherwise cost a scan of the whole state space per tree.
Eigen::Index MarkovChainAnalyzer::stateIndexForHash(uint64_t treeHash) const {

    std::unordered_map<uint64_t,Eigen::Index>::const_iterator it = hashToIndex.find(treeHash);
    if (it == hashToIndex.end())
        return -1;
    return it->second;
}

double MarkovChainAnalyzer::stationaryDiscrepancy(void) const {

    Vector piP;
    if (isSparse)
        piP = P_row_sparse.transpose() * pi;
    else
        piP = P_dense.transpose() * pi;
    return (piP - pi).norm();
}

double MarkovChainAnalyzer::stationaryVariance(const Vector& f) const {

    if (f.size() != static_cast<Eigen::Index>(n))
        return std::numeric_limits<double>::quiet_NaN();
    double mean = pi.dot(f);
    double m2 = (pi.array() * f.array() * f.array()).sum();
    return std::max(0.0, m2 - mean * mean);
}

MarkovChainAnalyzer::DenseMatrix MarkovChainAnalyzer::symmetrizedDense(void) const {

    DenseMatrix P = denseTransitionMatrix();
    Vector s    = pi.cwiseSqrt();
    Vector sinv = s.cwiseInverse();
    DenseMatrix S = s.asDiagonal() * P * sinv.asDiagonal();
    S = 0.5 * (S + S.transpose());
    return S;
}

MarkovChainAnalyzer::SparseMatrix MarkovChainAnalyzer::symmetrizedSparse(void) const {

    // S = D^{1/2} P D^{-1/2}, D = diag(pi). By detailed balance S is symmetric with the
    // same (real) spectrum as P. We symmetrize explicitly to absorb round-off so the
    // downstream solver sees an exactly symmetric operator.
    Vector s    = pi.cwiseSqrt();
    Vector sinv = s.cwiseInverse();
    SparseMatrix S = s.asDiagonal() * P_row_sparse * sinv.asDiagonal();
    SparseMatrix St = SparseMatrix(S.transpose());
    S = 0.5 * (S + St);
    S.makeCompressed();
    return S;
}

double MarkovChainAnalyzer::transitionProbability(Eigen::Index i, Eigen::Index j) const {

    return isSparse ? P_row_sparse.coeff(i, j) : P_dense(i, j);
}

bool MarkovChainAnalyzer::verifyStationary(double tol) const {

    Vector piP;
    if (isSparse)
        piP = P_row_sparse.transpose() * pi;
    else
        piP = P_dense.transpose() * pi;
    double err = (piP - pi).norm();
    std::ios::fmtflags savedFlags = std::cout.flags();
    std::streamsize savedPrec = std::cout.precision();
    std::cout << std::fixed << std::setprecision(20);
    std::cout << "   * Stationary verification discrepancy: " << err << "\n";
    std::cout.flags(savedFlags);
    std::cout.precision(savedPrec);
    return err < tol;
}

// Write the mean first-passage time to a given tree, normally the MAP tree, from every topology in
// the state space.
void MarkovChainAnalyzer::writeMeanFirstPassageTimesToTree(uint64_t treeHash, const std::string& fileName, int maxIterations, double tolerance) const {

    Eigen::Index target = stateIndexForHash(treeHash);
    if (target < 0)
        {
        if (!supportWasRestricted)
            std::cerr << "Warning: tree " << treeHash << " is not among the states of " << name
                  << "; no mean first-passage times were written.\n";
        return;
        }

    // The caller is expected to pass the MAP tree, which should also carry the largest posterior
    // probability. Disagreement would mean the two are being identified in different ways, so it is
    // worth reporting rather than silently accepting.
    Eigen::Index best = getMAPTreeIndex();
    if (best != target)
        std::cerr << "Warning: the target tree supplied to " << name << " is state " << target
                  << ", but the largest posterior probability belongs to state " << best << ".\n";

    Vector m = meanFirstPassageTimesToState(target, maxIterations, tolerance);
    if (m.size() == 0)
        return;

    std::ofstream out(fileName);
    if (!out)
        throw std::runtime_error("Could not open mean first-passage time file: " + fileName);

    out << "treeIndex\ttreeHash\tposteriorProbability\tmeanFirstPassageTime\n";
    out << std::setprecision(10);
    for (Eigen::Index i = 0; i < m.size(); ++i)
        {
        out << i << '\t' << stateHashes[static_cast<size_t>(i)] << '\t'
            << pi(i) << '\t' << m(i) << '\n';
        }

    // Two averages of the passage times are worth reporting, and they answer different questions.
    //
    // Averaging with equal weight on every topology gives the expected number of iterations needed
    // to reach the target from a tree drawn from the PRIOR, which is uniform over topologies here.
    // This is the quantity that matters in practice, because an MCMC analysis is started from a
    // tree drawn from the prior, and it therefore measures the length of the burn-in needed before
    // the chain first finds the best tree.
    //
    // Averaging with weight equal to the posterior probability of each topology instead gives the
    // expected time to reach the target from a tree drawn from the POSTERIOR, that is, from
    // stationarity. It describes how long a chain that has already converged typically goes between
    // visits to the target, and is not a statement about burn-in.
    //
    // The mean time to RETURN to the target requires no calculation at all: it is the reciprocal of
    // the target's posterior probability.
    std::cerr << "   Mean first-passage time to the MAP tree for " << name << ":\n"
              << "   * from a tree drawn from the prior (uniform start): " << m.mean() << " iterations\n"
              << "   * from a tree drawn from the posterior            : " << pi.dot(m) << " iterations\n"
              << "   * mean time to return to the MAP tree             : "
              << (pi(target) > 0.0 ? 1.0 / pi(target) : 0.0) << " iterations\n";
}

void MarkovChainAnalyzer::writePosteriorTsv(const std::string& fileName) const {

    std::ofstream out(fileName);
    if (!out)
        throw std::runtime_error("Could not open posterior file: " + fileName);

    out << std::setprecision(17);
    out << "state_index\ttree_hash\tposterior_probability\n";
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(n); ++i)
        {
        uint64_t h = (static_cast<size_t>(i) < stateHashes.size()) ? stateHashes[static_cast<size_t>(i)] : 0;
        out << i << "\t" << h << "\t" << pi(i) << "\n";
        }
}

void MarkovChainAnalyzer::writeTransitionKernelTsv(const std::string& fileName, bool denseFormat) const {

    std::ofstream out(fileName);
    if (!out)
        throw std::runtime_error("Could not open transition-kernel file: " + fileName);

    out << std::setprecision(17);

    if (denseFormat)
        {
        DenseMatrix P = denseTransitionMatrix();
        out << "state_index\ttree_hash";
        for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(n); ++j)
            out << "\tP_to_" << j;
        out << "\n";

        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(n); ++i)
            {
            uint64_t hi = (static_cast<size_t>(i) < stateHashes.size()) ? stateHashes[static_cast<size_t>(i)] : 0;
            out << i << "\t" << hi;
            for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(n); ++j)
                out << "\t" << P(i, j);
            out << "\n";
            }
        return;
        }

    // Coordinate format is much more practical for n = 10395:
    // one row per nonzero P_ij.
    out << "from_index\tfrom_hash\tto_index\tto_hash\tprobability\n";
    if (isSparse)
        {
        ensureRowSparse();
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(n); ++i)
            {
            uint64_t hi = (static_cast<size_t>(i) < stateHashes.size()) ? stateHashes[static_cast<size_t>(i)] : 0;
            for (RowSparseMatrix::InnerIterator it(P_row_sparse, i); it; ++it)
                {
                Eigen::Index j = it.col();
                uint64_t hj = (static_cast<size_t>(j) < stateHashes.size()) ? stateHashes[static_cast<size_t>(j)] : 0;
                out << i << "\t" << hi << "\t" << j << "\t" << hj << "\t" << it.value() << "\n";
                }
            }
        }
    else
        {
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(n); ++i)
            {
            uint64_t hi = (static_cast<size_t>(i) < stateHashes.size()) ? stateHashes[static_cast<size_t>(i)] : 0;
            for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(n); ++j)
                {
                double p = P_dense(i, j);
                if (p == 0.0)
                    continue;
                uint64_t hj = (static_cast<size_t>(j) < stateHashes.size()) ? stateHashes[static_cast<size_t>(j)] : 0;
                out << i << "\t" << hi << "\t" << j << "\t" << hj << "\t" << p << "\n";
                }
            }
        }
}

void MarkovChainAnalyzer::writeEfficiencyTsvHeader(std::ostream& os) {

    os << "moveType\tpower\tfunctional\tposteriorProbability\tindependentVariance"
          "\tasymptoticVariance\tefficiency\tintegratedAutocorrelationTime"
          "\tcgIterations\tcgRelativeResidual\tconverged\n";
}

void MarkovChainAnalyzer::writeEfficiencyTsvRow(std::ostream& os, const std::string& moveType, double power, const std::string& functional, const EfficiencyInfo& info) const {

    os << moveType << '\t' << power << '\t' << functional << '\t'
       << info.posterior_mean << '\t' << info.independent_variance << '\t'
       << info.asymptotic_variance << '\t' << info.efficiency << '\t'
       << info.integrated_autocorrelation_time << '\t'
       << info.iterations << '\t' << info.relative_residual << '\t'
       << (info.converged ? 1 : 0) << '\n';
}

void MarkovChainAnalyzer::writeFullEigenSystemTsv(const std::string& filePrefix, bool writeEigenvectors) const {

    // The MH kernel is reversible, so we diagonalize the symmetric S = D^{1/2} P D^{-1/2}.
    // Its eigenvalues are exactly those of P (real), and the pi-orthonormal right eigenvectors
    // of P are recovered as v_j = D^{-1/2} u_j. The imag columns are retained (== 0) so the
    // file schema matches earlier output. The eigensystem is the cached one shared with the other
    // dense routines (descending algebraic order, lambda_1 = 1 first).
    if (!ensureDenseEigensystem())
        throw std::runtime_error("Symmetric eigensystem calculation failed for: " + filePrefix);

    const Eigen::VectorXd& evals = denseEigenvalues;
    const DenseMatrix&     U     = denseEigenvectors;
    Eigen::Index N = evals.size();
    Vector sinv = pi.cwiseSqrt().cwiseInverse();

    // Residuals ||P v_k - lambda_k v_k|| / ||v_k||, v_k = D^{-1/2} u_k. One matrix-vector product per
    // eigenvector made this another O(n^3) pass done a column at a time; it is formed instead as a
    // matrix-matrix product over blocks of columns (multithreaded dgemm where LAPACK is available).
    Eigen::VectorXd residuals = Eigen::VectorXd::Constant(N, std::numeric_limits<double>::quiet_NaN());
    if (writeEigenvectors)
        {
        DenseMatrix P = denseTransitionMatrix();
        const Eigen::Index blockCols = 512;
        for (Eigen::Index c0 = 0; c0 < N; c0 += blockCols)
            {
            Eigen::Index nc = std::min(blockCols, N - c0);
            DenseMatrix Vb = sinv.asDiagonal() * U.middleCols(c0, nc);
            DenseMatrix PVb(N, nc);
#if MCA_USE_LAPACK
            int m = static_cast<int>(N), ncols = static_cast<int>(nc), kk = static_cast<int>(N);
            const double one = 1.0, zero = 0.0;
            const char noTrans = 'N';
            dgemm_(&noTrans, &noTrans, &m, &ncols, &kk, &one, P.data(), &m, Vb.data(), &kk, &zero, PVb.data(), &m);
#else
            PVb.noalias() = P * Vb;
#endif
            for (Eigen::Index j = 0; j < nc; ++j)
                {
                double denom = std::max(Vb.col(j).norm(), std::numeric_limits<double>::min());
                residuals(c0 + j) = (PVb.col(j) - evals(c0 + j) * Vb.col(j)).norm() / denom;
                }
            }
        }

    std::ofstream evalOut(filePrefix + ".eigenvalues.tsv");
    if (!evalOut)
        throw std::runtime_error("Could not open eigenvalue file: " + filePrefix + ".eigenvalues.tsv");

    evalOut << std::setprecision(17);
    evalOut << "eigen_index\treal\timag\tmodulus\tresidual\n";
    for (Eigen::Index k = 0; k < N; ++k)
        {
        evalOut << k << "\t" << evals(k) << "\t" << 0.0
                << "\t" << std::abs(evals(k)) << "\t" << residuals(k) << "\n";
        }

    if (!writeEigenvectors)
        return;

    std::ofstream vecOut(filePrefix + ".eigenvectors.tsv");
    if (!vecOut)
        throw std::runtime_error("Could not open eigenvector file: " + filePrefix + ".eigenvectors.tsv");

    vecOut << std::setprecision(17);
    vecOut << "eigen_index\tstate_index\ttree_hash\treal\timag\n";
    for (Eigen::Index k = 0; k < U.cols(); ++k)
        {
        for (Eigen::Index i = 0; i < U.rows(); ++i)
            {
            uint64_t h = (static_cast<size_t>(i) < stateHashes.size()) ? stateHashes[static_cast<size_t>(i)] : 0;
            vecOut << k << "\t" << i << "\t" << h << "\t"
                   << sinv(i) * U(i, k) << "\t" << 0.0 << "\n";
            }
        }
}

void MarkovChainAnalyzer::writeSmallHittingTimeFiles(const std::string& filePrefix, bool writeAllPairs) const {

    Eigen::Index mapIdx = getMAPTreeIndex();
    std::vector<Eigen::Index> targets95 = posteriorMassSet(0.95);

    Vector hMap = hittingTimesToStateVector(mapIdx);
    Vector h95 = hittingTimesToSetVector(targets95);

    std::ofstream out(filePrefix + ".hitting_times.tsv");
    if (!out)
        throw std::runtime_error("Could not open hitting-time file: " + filePrefix + ".hitting_times.tsv");

    out << std::setprecision(17);
    out << "state_index\ttree_hash\tposterior_probability\thit_map\thit_posterior95_set\n";
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(n); ++i)
        {
        uint64_t h = (static_cast<size_t>(i) < stateHashes.size()) ? stateHashes[static_cast<size_t>(i)] : 0;
        out << i << "\t" << h << "\t" << pi(i) << "\t"
            << (hMap.size() == static_cast<Eigen::Index>(n) ? hMap(i) : std::numeric_limits<double>::quiet_NaN()) << "\t"
            << (h95.size() == static_cast<Eigen::Index>(n) ? h95(i) : std::numeric_limits<double>::quiet_NaN()) << "\n";
        }

    std::ofstream summary(filePrefix + ".hitting_summary.tsv");
    if (!summary)
        throw std::runtime_error("Could not open hitting-time summary file: " + filePrefix + ".hitting_summary.tsv");

    double mass95 = 0.0;
    for (Eigen::Index idx : targets95)
        mass95 += pi(idx);

    summary << std::setprecision(17);
    summary << "map_index\tmap_hash\tmap_posterior\tposterior95_set_size\tposterior95_actual_mass\tmean_hit_map_uniform_start\tmean_hit_posterior95_uniform_start\n";
    uint64_t mapHash = (static_cast<size_t>(mapIdx) < stateHashes.size()) ? stateHashes[static_cast<size_t>(mapIdx)] : 0;
    summary << mapIdx << "\t" << mapHash << "\t" << pi(mapIdx) << "\t"
            << targets95.size() << "\t" << mass95 << "\t"
            << (hMap.size() == static_cast<Eigen::Index>(n) ? hMap.mean() : std::numeric_limits<double>::quiet_NaN()) << "\t"
            << (h95.size() == static_cast<Eigen::Index>(n) ? h95.mean() : std::numeric_limits<double>::quiet_NaN()) << "\n";

    if (!writeAllPairs)
        return;

    DenseMatrix M = computeHittingTimes();
    if (M.rows() == 0)
        return;

    std::ofstream allOut(filePrefix + ".all_pairs_hitting_times.tsv");
    if (!allOut)
        throw std::runtime_error("Could not open all-pairs hitting-time file: " + filePrefix + ".all_pairs_hitting_times.tsv");

    allOut << std::setprecision(17);
    allOut << "from_index\tfrom_hash\tto_index\tto_hash\thitting_time\n";
    for (Eigen::Index i = 0; i < M.rows(); ++i)
        {
        uint64_t hi = (static_cast<size_t>(i) < stateHashes.size()) ? stateHashes[static_cast<size_t>(i)] : 0;
        for (Eigen::Index j = 0; j < M.cols(); ++j)
            {
            uint64_t hj = (static_cast<size_t>(j) < stateHashes.size()) ? stateHashes[static_cast<size_t>(j)] : 0;
            allOut << i << "\t" << hi << "\t" << j << "\t" << hj << "\t" << M(i, j) << "\n";
            }
        }
}

void MarkovChainAnalyzer::writeSmallStateAnalysisFiles(const std::string& filePrefix, bool writeDenseKernel, bool writeFullEigenvectors, bool writeAllPairsHittingTimes) const {

    writePosteriorTsv(filePrefix + ".posterior.tsv");
    writeTransitionKernelTsv(filePrefix + ".transition_kernel.tsv", false);

    if (writeDenseKernel)
        writeTransitionKernelTsv(filePrefix + ".transition_kernel_dense.tsv", true);

    writeFullEigenSystemTsv(filePrefix, writeFullEigenvectors);
    writeSmallHittingTimeFiles(filePrefix, writeAllPairsHittingTimes);
}

void MarkovChainAnalyzer::writeTsvHeader(std::ostream& os) {

    os << "move_type"
       << "\tpower"
       << "\tanalysis_name"
       << "\tnum_states"
       << "\tstationary_discrepancy"
       << "\tirreducible_at_zero"
       << "\tlargest_threshold_irreducible"
       << "\tmin_positive_offdiag_transition"
       << "\tmax_offdiag_transition"
       << "\tmean_positive_offdiag_transition"
       << "\tmin_leave_probability"
       << "\tmax_leave_probability"
       << "\tmean_leave_probability"
       << "\tnum_positive_offdiag_transitions"
       << "\tnum_states_with_zero_leave_probability"
       << "\tirreducible_gt_1e_minus_16"
       << "\treachable_from0_frac_gt_1e_minus_16"
       << "\tcan_reach0_frac_gt_1e_minus_16"
       << "\ttransitions_le_1e_minus_16"
       << "\tstates_leave_le_1e_minus_16"
       << "\tirreducible_gt_1e_minus_14"
       << "\treachable_from0_frac_gt_1e_minus_14"
       << "\tcan_reach0_frac_gt_1e_minus_14"
       << "\ttransitions_le_1e_minus_14"
       << "\tstates_leave_le_1e_minus_14"
       << "\tirreducible_gt_1e_minus_12"
       << "\treachable_from0_frac_gt_1e_minus_12"
       << "\tcan_reach0_frac_gt_1e_minus_12"
       << "\ttransitions_le_1e_minus_12"
       << "\tstates_leave_le_1e_minus_12"
       << "\tirreducible_gt_1e_minus_10"
       << "\treachable_from0_frac_gt_1e_minus_10"
       << "\tcan_reach0_frac_gt_1e_minus_10"
       << "\ttransitions_le_1e_minus_10"
       << "\tstates_leave_le_1e_minus_10"
       << "\tirreducible_gt_1e_minus_8"
       << "\treachable_from0_frac_gt_1e_minus_8"
       << "\tcan_reach0_frac_gt_1e_minus_8"
       << "\ttransitions_le_1e_minus_8"
       << "\tstates_leave_le_1e_minus_8"
       << "\tirreducible_gt_1e_minus_6"
       << "\treachable_from0_frac_gt_1e_minus_6"
       << "\tcan_reach0_frac_gt_1e_minus_6"
       << "\ttransitions_le_1e_minus_6"
       << "\tstates_leave_le_1e_minus_6"
       << "\tdetailed_balance_checked"
       << "\tdetailed_balance_reversible"
       << "\tdetailed_balance_max_abs_error"
       << "\tdetailed_balance_max_relative_error"
       << "\tslem_abs_lambda2"
       << "\tspectral_gap"
       << "\trelaxation_time"
       << "\tworst_case_iact"
       << "\tsparse_eigen_calculation"
       << "\teigenvalues_converged_or_computed"
       << "\teigenvalue_1_multiplicity_among_computed"
       << "\thas_complex_eigenvalues"
       << "\taverage_acceptance_rate"
       << "\tentropy_rate_nats_per_step"
       << "\tkemeny_estimate"
       << "\tkemeny_is_exact"
       << "\tposterior_sweep_conductance"
       << "\teigenvector_sweep_conductance"
       << "\tmean_return_time_min"
       << "\tmean_return_time_max"
       << "\tmap_index"
       << "\tmap_posterior"
       << "\tposterior95_set_size"
       << "\tposterior95_actual_mass"
       << "\tmean_hitting_time_map_uniform_start"
       << "\tmean_hitting_time_posterior95_uniform_start"
       << "\tmixing_time_upper_bound_epsilon"
       << "\tmixing_time_upper_bound"
       << "\tlambda2_algebraic"
       << "\tlambda_min"
       << "\tiact_map_indicator"
       << "\tiact_posterior95_indicator"
       << "\tspectral_valid"
       << "\tmax_eigen_residual"
       << "\tspectral_status"
       << "\n";
}

void MarkovChainAnalyzer::writeTsvRow(std::ostream& os, const std::string& moveType, double power, double epsilon) const {

    os << std::setprecision(17);

    double statErr = stationaryDiscrepancy();

    DetailedBalanceInfo db;
    bool dbChecked = (n <= detailedBalanceStateLimit);
    if (dbChecked)
        db = computeDetailedBalanceInfo();
    else
        db.skipped = true;

    std::vector<double> nearThresholds = defaultIrreducibilityThresholds();
    ThresholdedIrreducibilityInfo irrSummary = computeThresholdedIrreducibilityInfo(nearThresholds);
    TransitionProbabilityInfo transInfo = computeTransitionProbabilityInfo(nearThresholds);

    SpectralInfo spec = getSpectralInfo(defaultSparseEigenvalues);
    double accept = averageAcceptanceRate();
    double entropy = entropyRate();

    bool kemenyExact = (n <= denseStateLimit);
    double kemeny = kemenyExact ? kemenyConstant() : approximateKemenyConstant(32);

    double postCond = posteriorSweepConductance();
    double eigCond = eigenvectorSweepConductance();

    Vector mrt = meanReturnTimes();
    double mrtMin = mrt.minCoeff();
    double mrtMax = mrt.maxCoeff();

    Eigen::Index mapIdx = getMAPTreeIndex();
    double mapPost = pi(mapIdx);

    std::vector<Eigen::Index> targets95 = posteriorMassSet(0.95);
    double mass95 = 0.0;
    for (Eigen::Index idx : targets95)
        mass95 += pi(idx);

    double hitMap = meanHittingTimeToStateQuiet(mapIdx);
    double hit95 = meanHittingTimeToSetQuiet(targets95);
    double mixBound = mixingTimeUpperBoundValue(epsilon);

    // Exact functional IACTs from Poisson solves (efficiencyFor), not the truncated leading-mode
    // spectral estimator: for n > denseStateLimit that estimator collapses onto the worst-case IACT
    // for indicators that concentrate on the slow subspace, whatever their true slow-mode overlap.
    // The solve is exact whenever it converges; a non-converged solve is reported as NaN rather than
    // a plausible-looking wrong number.
    EfficiencyInfo effMap = efficiencyFor(indicatorMAP());
    EfficiencyInfo eff95  = efficiencyFor(indicatorPosteriorMassSet(0.95));
    double iactMap = effMap.converged ? effMap.integrated_autocorrelation_time : std::numeric_limits<double>::quiet_NaN();
    double iact95  = eff95.converged  ? eff95.integrated_autocorrelation_time : std::numeric_limits<double>::quiet_NaN();

    os << moveType
       << "\t" << power
       << "\t" << name
       << "\t" << n
       << "\t" << statErr
       << "\t" << (irrSummary.irreducible_at_zero ? 1 : 0)
       << "\t" << irrSummary.largest_threshold_irreducible
       << "\t" << transInfo.min_positive_offdiag_transition
       << "\t" << transInfo.max_offdiag_transition
       << "\t" << transInfo.mean_positive_offdiag_transition
       << "\t" << transInfo.min_leave_probability
       << "\t" << transInfo.max_leave_probability
       << "\t" << transInfo.mean_leave_probability
       << "\t" << transInfo.num_positive_offdiag_transitions
       << "\t" << transInfo.num_states_with_zero_leave_probability
       << "\t" << (irrSummary.results.size() > 1 && irrSummary.results[1].irreducible ? 1 : 0)
       << "\t" << (irrSummary.results.size() > 1 && irrSummary.results[1].num_states > 0 ? static_cast<double>(irrSummary.results[1].states_reachable_from_0) / static_cast<double>(irrSummary.results[1].num_states) : std::numeric_limits<double>::quiet_NaN())
       << "\t" << (irrSummary.results.size() > 1 && irrSummary.results[1].num_states > 0 ? static_cast<double>(irrSummary.results[1].states_that_can_reach_0) / static_cast<double>(irrSummary.results[1].num_states) : std::numeric_limits<double>::quiet_NaN())
       << "\t" << (transInfo.num_transitions_le_threshold.size() > 1 ? transInfo.num_transitions_le_threshold[1] : 0)
       << "\t" << (transInfo.num_states_leave_le_threshold.size() > 1 ? transInfo.num_states_leave_le_threshold[1] : 0)
       << "\t" << (irrSummary.results.size() > 2 && irrSummary.results[2].irreducible ? 1 : 0)
       << "\t" << (irrSummary.results.size() > 2 && irrSummary.results[2].num_states > 0 ? static_cast<double>(irrSummary.results[2].states_reachable_from_0) / static_cast<double>(irrSummary.results[2].num_states) : std::numeric_limits<double>::quiet_NaN())
       << "\t" << (irrSummary.results.size() > 2 && irrSummary.results[2].num_states > 0 ? static_cast<double>(irrSummary.results[2].states_that_can_reach_0) / static_cast<double>(irrSummary.results[2].num_states) : std::numeric_limits<double>::quiet_NaN())
       << "\t" << (transInfo.num_transitions_le_threshold.size() > 2 ? transInfo.num_transitions_le_threshold[2] : 0)
       << "\t" << (transInfo.num_states_leave_le_threshold.size() > 2 ? transInfo.num_states_leave_le_threshold[2] : 0)
       << "\t" << (irrSummary.results.size() > 3 && irrSummary.results[3].irreducible ? 1 : 0)
       << "\t" << (irrSummary.results.size() > 3 && irrSummary.results[3].num_states > 0 ? static_cast<double>(irrSummary.results[3].states_reachable_from_0) / static_cast<double>(irrSummary.results[3].num_states) : std::numeric_limits<double>::quiet_NaN())
       << "\t" << (irrSummary.results.size() > 3 && irrSummary.results[3].num_states > 0 ? static_cast<double>(irrSummary.results[3].states_that_can_reach_0) / static_cast<double>(irrSummary.results[3].num_states) : std::numeric_limits<double>::quiet_NaN())
       << "\t" << (transInfo.num_transitions_le_threshold.size() > 3 ? transInfo.num_transitions_le_threshold[3] : 0)
       << "\t" << (transInfo.num_states_leave_le_threshold.size() > 3 ? transInfo.num_states_leave_le_threshold[3] : 0)
       << "\t" << (irrSummary.results.size() > 4 && irrSummary.results[4].irreducible ? 1 : 0)
       << "\t" << (irrSummary.results.size() > 4 && irrSummary.results[4].num_states > 0 ? static_cast<double>(irrSummary.results[4].states_reachable_from_0) / static_cast<double>(irrSummary.results[4].num_states) : std::numeric_limits<double>::quiet_NaN())
       << "\t" << (irrSummary.results.size() > 4 && irrSummary.results[4].num_states > 0 ? static_cast<double>(irrSummary.results[4].states_that_can_reach_0) / static_cast<double>(irrSummary.results[4].num_states) : std::numeric_limits<double>::quiet_NaN())
       << "\t" << (transInfo.num_transitions_le_threshold.size() > 4 ? transInfo.num_transitions_le_threshold[4] : 0)
       << "\t" << (transInfo.num_states_leave_le_threshold.size() > 4 ? transInfo.num_states_leave_le_threshold[4] : 0)
       << "\t" << (irrSummary.results.size() > 5 && irrSummary.results[5].irreducible ? 1 : 0)
       << "\t" << (irrSummary.results.size() > 5 && irrSummary.results[5].num_states > 0 ? static_cast<double>(irrSummary.results[5].states_reachable_from_0) / static_cast<double>(irrSummary.results[5].num_states) : std::numeric_limits<double>::quiet_NaN())
       << "\t" << (irrSummary.results.size() > 5 && irrSummary.results[5].num_states > 0 ? static_cast<double>(irrSummary.results[5].states_that_can_reach_0) / static_cast<double>(irrSummary.results[5].num_states) : std::numeric_limits<double>::quiet_NaN())
       << "\t" << (transInfo.num_transitions_le_threshold.size() > 5 ? transInfo.num_transitions_le_threshold[5] : 0)
       << "\t" << (transInfo.num_states_leave_le_threshold.size() > 5 ? transInfo.num_states_leave_le_threshold[5] : 0)
       << "\t" << (irrSummary.results.size() > 6 && irrSummary.results[6].irreducible ? 1 : 0)
       << "\t" << (irrSummary.results.size() > 6 && irrSummary.results[6].num_states > 0 ? static_cast<double>(irrSummary.results[6].states_reachable_from_0) / static_cast<double>(irrSummary.results[6].num_states) : std::numeric_limits<double>::quiet_NaN())
       << "\t" << (irrSummary.results.size() > 6 && irrSummary.results[6].num_states > 0 ? static_cast<double>(irrSummary.results[6].states_that_can_reach_0) / static_cast<double>(irrSummary.results[6].num_states) : std::numeric_limits<double>::quiet_NaN())
       << "\t" << (transInfo.num_transitions_le_threshold.size() > 6 ? transInfo.num_transitions_le_threshold[6] : 0)
       << "\t" << (transInfo.num_states_leave_le_threshold.size() > 6 ? transInfo.num_states_leave_le_threshold[6] : 0)
       << "\t" << (dbChecked ? 1 : 0)
       << "\t" << (dbChecked ? (db.reversible ? 1.0 : 0.0) : std::numeric_limits<double>::quiet_NaN())
       << "\t" << (dbChecked ? db.max_abs_error : std::numeric_limits<double>::quiet_NaN())
       << "\t" << (dbChecked ? db.max_relative_error : std::numeric_limits<double>::quiet_NaN())
       << "\t" << spec.lambda2_abs
       << "\t" << spec.spectral_gap
       << "\t" << spec.relaxation_time
       << "\t" << spec.worst_case_iact
       << "\t" << (spec.computed_sparse ? 1 : 0)
       << "\t" << spec.n_converged
       << "\t" << spec.multiplicity_of_1
       << "\t" << (spec.has_complex_eigenvalues ? 1 : 0)
       << "\t" << accept
       << "\t" << entropy
       << "\t" << kemeny
       << "\t" << (kemenyExact ? 1 : 0)
       << "\t" << postCond
       << "\t" << eigCond
       << "\t" << mrtMin
       << "\t" << mrtMax
       << "\t" << mapIdx
       << "\t" << mapPost
       << "\t" << targets95.size()
       << "\t" << mass95
       << "\t" << hitMap
       << "\t" << hit95
       << "\t" << epsilon
       << "\t" << mixBound
       << "\t" << spec.lambda2_algebraic
       << "\t" << spec.lambda_min
       << "\t" << iactMap
       << "\t" << iact95
       << "\t" << (spec.spectral_valid ? 1 : 0)
       << "\t" << spec.max_eigen_residual
       << "\t" << spec.spectral_status
       << "\n";
}

