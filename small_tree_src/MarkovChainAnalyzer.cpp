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
#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <limits>
#include <numeric>
#include <random>
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
    : threadPool(tp), P_sparse(P), pi(pi), n(P.rows()), isSparse(true) {
    
    P_sparse.makeCompressed();
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
                                                     Eigen::Index lo, Eigen::Index hi) :
                                        outerIndex(out), innerIndex(in), values(v),
                                        xIn(x), yOut(y), begin(lo), end(hi) { }
            void                    run(void) override;

        private:
            const int*              outerIndex;
            const int*              innerIndex;
            const double*           values;
            const double*           xIn;
            double*                 yOut;
            Eigen::Index            begin;
            Eigen::Index            end;
    };

    void SparseMatVecTask::run(void) {

        for (Eigen::Index i = begin; i < end; ++i)
            {
            double sum = 0.0;
            for (int k = outerIndex[i]; k < outerIndex[i+1]; ++k)
                sum += values[k] * xIn[innerIndex[k]];
            yOut[i] = sum;
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
    class AnalyzerSymOp {

        public:
            using Scalar = double;
                                    AnalyzerSymOp(void) = delete;
                                    AnalyzerSymOp(const MarkovChainAnalyzer* a, Eigen::Index dim) : self(a), n(dim) { }
            Eigen::Index            rows(void) const { return n; }
            Eigen::Index            cols(void) const { return n; }
            void                    perform_op(const double* x_in, double* y_out) const;

        private:
            const MarkovChainAnalyzer* self;
            Eigen::Index               n;
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

} // anonymous namespace

void AnalyzerSymOp::perform_op(const double* x_in, double* y_out) const {

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
    Eigen::Map<const Vector> xm(x, N);
    Vector t = xm.array() / piSqrtCached.array();
    Vector u(N);
    if (isSparse)
        sparseMatVec(t, u);
    else
        u.noalias() = P_dense * t;
    Eigen::Map<Vector>(y, N) = piSqrtCached.cwiseProduct(u);
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
    for (int col = 0; col < P_sparse.outerSize(); ++col)
        for (SparseMatrix::InnerIterator it(P_sparse, col); it; ++it)
            if (it.row() <= it.col())
                ++cnt;

    std::vector<int>    rows; rows.reserve(static_cast<size_t>(cnt));
    std::vector<int>    cols; cols.reserve(static_cast<size_t>(cnt));
    std::vector<double> vals; vals.reserve(static_cast<size_t>(cnt));
    for (int col = 0; col < P_sparse.outerSize(); ++col)
        for (SparseMatrix::InnerIterator it(P_sparse, col); it; ++it)
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
            acc += pi(i) * (1.0 - P_sparse.coeff(i, i));
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
        Vector diag = P_sparse.diagonal();
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
                for (TreeInfo* nb : cur->neighbors)
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

    pi = Vector::Zero(static_cast<Eigen::Index>(n));

    // Reserve exactly the number of entries the kernel will hold. Guessing low is expensive here:
    // a TBR kernel for ten taxa has some 590 million non-zero entries, and growing the triplet
    // array to that size from a small reservation copies tens of gigabytes.
    size_t numEntries = n;                                        // one diagonal entry per tree
    for (const auto& [h, info] : tcache)
        {
        if (info)
            numEntries += info->neighbors.size();
        }
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(numEntries);

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
    double worstProposalDefect = 0.0;

    for (const auto& [h_from, info_from] : tcache) 
        {
        if (!info_from || info_from->stateIndex < 0)
            continue;
        Eigen::Index i = static_cast<Eigen::Index>(info_from->stateIndex);
        pi(i) = info_from->posteriorProbability;
        double sumTransition = 0.0;

        double rawProposalMass = 0.0;
        for (double q : info_from->neighborProposalProbabilities)
            rawProposalMass += q;
        worstProposalDefect = std::max(worstProposalDefect, std::abs(rawProposalMass - 1.0));

        double pi_i   = info_from->posteriorProbability;
        double lnL_i  = info_from->lnLikelihood;
        size_t numQ   = info_from->neighborProposalProbabilities.size();

        for (size_t k = 0; k < info_from->neighbors.size(); ++k) 
            {
            TreeInfo* info_to = info_from->neighbors[k];
            if (!info_to || info_to->stateIndex < 0)
                continue;
            Eigen::Index j = static_cast<Eigen::Index>(info_to->stateIndex);

            double q_ij = (k < numQ) ? info_from->neighborProposalProbabilities[k] : 0.0;
            if (q_ij <= 0.0)
                continue;

            double q_ji = std::exp((lnL_i - info_to->neighborMaxLnL) * info_to->neighborProposalPower)
                        * info_to->neighborProposalNormInv;

            double pi_j = info_to->posteriorProbability;
            if (pi_i <= 0.0 || q_ji <= 0.0) 
                continue;

            double ratio      = (pi_j * q_ji) / (pi_i * q_ij);
            double alpha      = std::min(1.0, ratio);
            double trans_prob = q_ij * alpha;

            if (trans_prob > 0.0) 
                {
                triplets.emplace_back(i, j, trans_prob);
                sumTransition += trans_prob;
                }
            }

        double selfProb = 1.0 - sumTransition;
        if (selfProb < 0.0 && selfProb > -1e-12)
            selfProb = 0.0;
        if (selfProb < 0.0)
            std::cerr << "Warning: negative self transition at row " << i << ": " << selfProb << "\n";
        triplets.emplace_back(i, i, std::max(0.0, selfProb));
        }

    if (worstProposalDefect > 1e-6)
        std::cerr << "Warning: proposal probabilities deviate from 1 by up to "
                  << worstProposalDefect << "; the acceptance-rate interpretation of "
                     "1 - P_ii assumes a normalized proposal with no explicit self-proposal.\n";

    if (isSparse) 
        {
        P_sparse = SparseMatrix(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
        P_sparse.setFromTriplets(triplets.begin(), triplets.end());
        P_sparse.makeCompressed();
        rowSparseReady = false;
        } 
    else 
        {
        P_dense = DenseMatrix::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
        for (const auto& t : triplets)
            P_dense(t.row(), t.col()) += t.value();
        }
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
                double b = pi(j) * P_sparse.coeff(j, i);
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
        ensureRowSparse();
        for (Eigen::Index i = 0; i < N; ++i) 
            {
            size_t outDegree = 0;
            double leaveProb = 0.0;
            for (RowSparseMatrix::InnerIterator it(P_row_sparse, i); it; ++it) 
                {
                Eigen::Index j = it.col();
                double p = it.value();
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

    info.states_reachable_from_0 = countReachableForward(0, threshold);
    info.states_that_can_reach_0 = countReachableReverse(0, threshold);
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

    for (iterations = 0; iterations < maxIterations; ++iterations)
        {
        applyA(p, Ap);
        double pAp = p.dot(Ap);
        if (!(pAp > 0.0))
            break;                                                // positive definiteness lost
        double alpha = rz / pAp;
        x += alpha * p;
        r -= alpha * Ap;

        relativeResidual = r.norm() / bNorm;
        if (relativeResidual < tolerance)
            {
            ++iterations;
            return true;
            }

        z = r.array() / diagonal.array();
        double rzNext = r.dot(z);
        p  = z + (rzNext / rz) * p;
        rz = rzNext;
        }

    relativeResidual = r.norm() / bNorm;
    return relativeResidual < tolerance;
}

size_t MarkovChainAnalyzer::countReachableForward(Eigen::Index start, double threshold) const {

    Eigen::Index N = static_cast<Eigen::Index>(n);
    if (start < 0 || start >= N)
        return 0;

    std::vector<char> visited(static_cast<size_t>(N), 0);
    std::vector<Eigen::Index> stack;
    stack.reserve(1024);
    visited[static_cast<size_t>(start)] = 1;
    stack.push_back(start);
    size_t count = 1;

    if (isSparse) 
        {
        ensureRowSparse();
        while (!stack.empty()) 
            {
            Eigen::Index i = stack.back();
            stack.pop_back();
            for (RowSparseMatrix::InnerIterator it(P_row_sparse, i); it; ++it) 
                {
                Eigen::Index j = it.col();
                if (j == i) continue;
                if (it.value() <= threshold) continue;
                size_t jj = static_cast<size_t>(j);
                if (!visited[jj]) 
                    {
                    visited[jj] = 1;
                    stack.push_back(j);
                    ++count;
                    }
                }
            }
        }
    else 
        {
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
        }
    return count;
}

size_t MarkovChainAnalyzer::countReachableReverse(Eigen::Index start, double threshold) const {

    Eigen::Index N = static_cast<Eigen::Index>(n);
    if (start < 0 || start >= N)
        return 0;

    std::vector<char> visited(static_cast<size_t>(N), 0);
    std::vector<Eigen::Index> stack;
    stack.reserve(1024);
    visited[static_cast<size_t>(start)] = 1;
    stack.push_back(start);
    size_t count = 1;

    if (isSparse) 
        {
        // P_sparse is column-major. Iterating over column i gives all j such that P(j,i) > 0,
        // which are exactly the outgoing neighbors of i in the reversed transition graph.
        while (!stack.empty()) 
            {
            Eigen::Index i = stack.back();
            stack.pop_back();
            for (SparseMatrix::InnerIterator it(P_sparse, i); it; ++it) 
                {
                Eigen::Index j = it.row();
                if (j == i) continue;
                if (it.value() <= threshold) continue;
                size_t jj = static_cast<size_t>(j);
                if (!visited[jj]) 
                    {
                    visited[jj] = 1;
                    stack.push_back(j);
                    ++count;
                    }
                }
            }
        }
    else 
        {
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
    DenseMatrix Pmat = isSparse ? DenseMatrix(P_sparse) : P_dense;
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
    DenseMatrix S = symmetrizedDense();
    Eigen::SelfAdjointEigenSolver<DenseMatrix> es(S);
    if (es.info() != Eigen::Success)
        {
        info.spectral_status = "dense_selfadjoint_failed";
        return info;
        }

    info.eigenvalues_real = es.eigenvalues().reverse();   // descending algebraic order
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

    int kBot = std::min(2, static_cast<int>(n) - 1);

    // Attempts widen the Krylov subspace and iteration budget; the last is deliberately generous, since
    // resolving lambda_2 at the top of a dense near-1 cluster needs ncv well above the cluster size.
    struct SolverAttempt { int ncv; int maxit; double tol; };
    std::vector<SolverAttempt> attempts;
    attempts.push_back({std::min(static_cast<int>(n), std::max(2 * nev + 1,  20)),  2000, 1e-10});
    attempts.push_back({std::min(static_cast<int>(n), std::max(6 * nev + 40, 120)),  6000, 1e-11});
    attempts.push_back({std::min(static_cast<int>(n), std::max(12 * nev + 80, 300)), 12000, 1e-12});
    attempts.push_back({std::min(static_cast<int>(n), std::max(24 * nev + 160, 700)),24000, 1e-12});

    double bestResidual = std::numeric_limits<double>::infinity();
    int attemptNumber = 0;

    for (const SolverAttempt& attempt : attempts)
        {
        attemptNumber++;

        int ncvTop = std::max(attempt.ncv, nev + 2);
        ncvTop = std::min(ncvTop, static_cast<int>(n));
        int ncvBot = std::min(static_cast<int>(n), std::max(std::max(2 * kBot + 1, 20), kBot + 2));

        // lambda_2 = largest eigenvalue of the deflated operator; lambda_min from the base operator.
        Spectra::SymEigsSolver<DeflatedSymOp> eigsTop(dop, nev, ncvTop);
        Spectra::SymEigsSolver<Op>            eigsBot(op, kBot, ncvBot);

        int nconvTop = 0, nconvBot = 0;
        try
            {
            eigsTop.init();
            nconvTop = (int)eigsTop.compute(Spectra::SortRule::LargestAlge, attempt.maxit, attempt.tol);
            eigsBot.init();
            nconvBot = (int)eigsBot.compute(Spectra::SortRule::SmallestAlge, attempt.maxit, attempt.tol);
            }
        catch (const std::exception& e)
            {
            std::cerr << "Note: eigensolver attempt " << attemptNumber << " for " << name
                      << " failed internally (" << e.what() << "); retrying with a larger subspace.\n";
            continue;
            }

        SpectralInfo info;
        info.computed_sparse = true;
        info.n_converged = std::max(0, nconvTop) + std::max(0, nconvBot);
        info.ncv_used = ncvTop;
        info.max_iterations_used = attempt.maxit;
        info.tolerance_used = attempt.tol;
        info.num_solver_attempts = attemptNumber;
        info.spectral_status = "insufficient_converged_eigenvalues";

        // Need at least lambda_2 from the deflated top solve.
        if (nconvTop < 1 || eigsTop.info() != Spectra::CompInfo::Successful)
            {
            if (info.n_converged > bestInfo.n_converged)
                bestInfo = info;
            continue;
            }

        // Assemble the spectrum: the known unit eigenvalue, the deflated top eigenvalues (lambda_2, ...),
        // and the most-negative eigenvalues. Deflated eigenvectors are eigenvectors of S with the same
        // eigenvalue (they are orthogonal to w), so residuals are measured against S directly.
        std::vector<std::pair<double, Eigen::VectorXd>> pairs;
        pairs.push_back({1.0, w});
        bool sawNaN = false;
        {
            Eigen::VectorXd vt = eigsTop.eigenvalues();
            Eigen::MatrixXd Vt = eigsTop.eigenvectors();
            for (Eigen::Index i = 0; i < vt.size(); ++i)
                {
                if (!std::isfinite(vt(i))) { sawNaN = true; continue; }
                // Skip the deflated unit mode: its eigenvector is w (eigenvalue mapped 1 -> 0), which is
                // not an eigenvector of S at that value. Genuine eigenvectors (lambda != 1) are
                // orthogonal to w, so a large overlap identifies the deflated direction.
                if (std::abs(w.dot(Vt.col(i))) > 0.5)
                    continue;
                pairs.push_back({vt(i), Vt.col(i)});
                }
            if (nconvBot > 0 && eigsBot.info() == Spectra::CompInfo::Successful)
                {
                Eigen::VectorXd vb = eigsBot.eigenvalues();
                Eigen::MatrixXd Vb = eigsBot.eigenvectors();
                for (Eigen::Index i = 0; i < vb.size(); ++i)
                    {
                    if (!std::isfinite(vb(i))) { sawNaN = true; continue; }
                    pairs.push_back({vb(i), Vb.col(i)});
                    }
                }
        }

        // A NaN eigenvalue from the solver means this subspace was too small; escalate.
        if (sawNaN && attemptNumber < static_cast<int>(attempts.size()))
            {
            if (info.n_converged > bestInfo.n_converged)
                bestInfo = info;
            continue;
            }

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
        double maxResidual = lambda1Residual;
        for (auto& pr : uniq)
            {
            if (std::abs(pr.first - 1.0) < 1e-12)
                continue;
            const Eigen::VectorXd& u = pr.second;
            double denom = std::max(u.norm(), std::numeric_limits<double>::min());
            Eigen::VectorXd Su(u.size());
            op.perform_op(u.data(), Su.data());
            double resid = (Su - pr.first * u).norm() / denom;
            maxResidual = std::max(maxResidual, resid);
            }
        info.max_eigen_residual = maxResidual;
        info.lambda1_error = lambda1Residual;          // deterministic, from the known eigenpair

        bool enough     = (m >= 2);
        bool lambda1OK  = std::isfinite(info.lambda1_error) && info.lambda1_error < 1e-5;
        bool residualOK = std::isfinite(info.max_eigen_residual) && info.max_eigen_residual < 1e-5;
        bool lambda2OK  = std::isfinite(info.lambda2_abs) && info.lambda2_abs <= 1.0 + 1e-7;

        info.spectral_valid = enough && lambda1OK && residualOK && lambda2OK;
        if (info.spectral_valid)
            {
            info.spectral_status = "ok";
            return info;
            }

        if (!lambda1OK)
            info.spectral_status = "dominant_eigenvalue_not_close_to_one";
        else if (!residualOK)
            info.spectral_status = "large_eigen_residual";
        else if (!lambda2OK)
            info.spectral_status = "lambda2_modulus_greater_than_one";
        else
            info.spectral_status = "invalid_unknown_reason";

        if (maxResidual < bestResidual || bestInfo.n_converged < 2)
            {
            bestResidual = maxResidual;
            bestInfo = info;
            }
        }

    if (!bestInfo.spectral_valid) 
        {
        std::cerr << "Warning: sparse eigen calculation unreliable for " << name
                  << " (status=" << bestInfo.spectral_status
                  << ", converged=" << bestInfo.n_converged
                  << ", max residual=" << bestInfo.max_eigen_residual
                  << ", lambda1 error=" << bestInfo.lambda1_error
                  << ").\n";
        bestInfo.spectral_gap = quietNaNValue();
        bestInfo.lambda2_abs = quietNaNValue();
        bestInfo.relaxation_time = quietNaNValue();
        bestInfo.worst_case_iact = quietNaNValue();
        }

    return bestInfo;
}

ThresholdedIrreducibilityInfo MarkovChainAnalyzer::computeThresholdedIrreducibilityInfo(const std::vector<double>& thresholds, double tinyLeaveProb) const {

    ThresholdedIrreducibilityInfo summary;
    summary.thresholds = thresholds;
    summary.results.reserve(thresholds.size());

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

            // Remove old A -> x contributions, because x moves from A^c to A.
            for (SparseMatrix::InnerIterator it(P_sparse, x); it; ++it) 
                {
                Eigen::Index i = it.row();
                if (inSet[static_cast<size_t>(i)])
                    flow -= pi(i) * it.value();
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

    return isSparse ? DenseMatrix(P_sparse) : P_dense;
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

    Vector diagP = isSparse ? Vector(P_sparse.diagonal()) : Vector(P_dense.diagonal());
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

    DenseMatrix P = isSparse ? DenseMatrix(P_sparse) : P_dense;
    Eigen::EigenSolver<DenseMatrix> solver(P);
    Eigen::VectorXcd evals = solver.eigenvalues();
    Eigen::MatrixXcd evecs = solver.eigenvectors();

    std::vector<std::pair<double, Eigen::Index>> moduli;
    moduli.reserve(static_cast<size_t>(n));
    for (Eigen::Index i = 0; i < evals.size(); ++i)
        moduli.push_back({std::abs(evals(i)), i});
    std::sort(moduli.begin(), moduli.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    if (moduli.size() < 2)
        return std::numeric_limits<double>::quiet_NaN();

    Eigen::Index eigIndex = moduli[1].second;
    std::vector<Eigen::Index> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&evecs, eigIndex](Eigen::Index a, Eigen::Index b) {
        return evecs(a, eigIndex).real() < evecs(b, eigIndex).real();
    });
    return conductanceFromOrderingFast(order);
}

void MarkovChainAnalyzer::ensureRowSparse(void) const {

    if (!isSparse || rowSparseReady)
        return;
    P_row_sparse = P_sparse;
    P_row_sparse.makeCompressed();
    rowSparseReady = true;
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

    if (n < 1)
        return false;
    DenseMatrix S = symmetrizedDense();
    Eigen::SelfAdjointEigenSolver<DenseMatrix> es(S);
    if (es.info() != Eigen::Success)
        return false;
    evals = es.eigenvalues();   // ascending
    U = es.eigenvectors();      // columns are Euclidean-orthonormal eigenvectors of S
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
    // Stochastic Lanczos quadrature: for Rademacher probes z deflated against w, Hutchinson gives
    // E[ z'^T f(M) z' ] = trace( f(M) (I - w w^T) ) with f(mu) = 1/mu, and each quadratic form is
    // approximated by Gauss quadrature read off an m-step Lanczos tridiagonalization of M.
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

    int m = std::max(2, std::min(lanczosSteps, static_cast<int>(N) - 1));
    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<int> coin(0, 1);

    auto applyM = [&](const Vector& x) -> Vector {
        Vector Sx(x.size());
        applyS(x.data(), Sx.data());
        return x - Sx;
    };

    double traceEstimate = 0.0;
    int validProbes = 0;

    for (int p = 0; p < numProbes; ++p)
        {
        Vector z(N);
        for (Eigen::Index i = 0; i < N; ++i)
            z(i) = coin(gen) ? 1.0 : -1.0;
        z -= (w.dot(z)) * w;                          // deflate the null space
        double z2 = z.squaredNorm();
        if (z2 <= 0.0)
            continue;

        // m-step Lanczos on M with full reorthogonalization (against the basis and w).
        std::vector<Vector> V;
        V.reserve(static_cast<size_t>(m));
        std::vector<double> alpha, beta;             // beta holds sub-diagonal entries
        Vector q = z / std::sqrt(z2);
        Vector qPrev = Vector::Zero(N);
        double betaPrev = 0.0;
        int steps = 0;

        for (int k = 0; k < m; ++k)
            {
            q -= (w.dot(q)) * w;
            double qn = q.norm();
            if (qn <= 1e-14)
                break;
            q /= qn;
            V.push_back(q);

            Vector Aq = applyM(q);
            double a = q.dot(Aq);
            alpha.push_back(a);

            Vector r = Aq - a * q - betaPrev * qPrev;
            r -= (w.dot(r)) * w;
            for (const Vector& v : V)
                r -= (v.dot(r)) * v;                  // full reorthogonalization

            double b = r.norm();
            ++steps;
            if (b <= 1e-14 || k == m - 1)
                break;
            beta.push_back(b);
            qPrev = q;
            betaPrev = b;
            q = r / b;
            }

        if (steps < 1)
            continue;

        Eigen::MatrixXd T = Eigen::MatrixXd::Zero(steps, steps);
        for (int k = 0; k < steps; ++k)
            {
            T(k, k) = alpha[static_cast<size_t>(k)];
            if (k + 1 < steps)
                {
                T(k, k + 1) = beta[static_cast<size_t>(k)];
                T(k + 1, k) = beta[static_cast<size_t>(k)];
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
            if (th > 1e-12)                           // skip the (deflated) null direction
                quad += tau0 * tau0 / th;
            }

        traceEstimate += z2 * quad;
        ++validProbes;
        }

    if (validProbes == 0)
        return std::numeric_limits<double>::quiet_NaN();
    double K = traceEstimate / static_cast<double>(validProbes);
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
        eigs.compute(rule, 6000, 1e-12);
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

    Vector diagP = isSparse ? Vector(P_sparse.diagonal()) : Vector(P_dense.diagonal());
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
        triplets.reserve(static_cast<size_t>(P_sparse.nonZeros() + N));
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
        triplets.reserve(static_cast<size_t>(P_sparse.nonZeros() + N));
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
        triplets.reserve(static_cast<size_t>(P_sparse.nonZeros() + N));
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
        Eigen::VectorXd ea;
        DenseMatrix Ua;
        if (!fullSymmetricEigensystem(ea, Ua))
            return false;
        Eigen::Index m = ea.size();
        evals.resize(m);
        U.resize(Ua.rows(), m);
        for (Eigen::Index i = 0; i < m; ++i)
            {
            evals(i) = ea(m - 1 - i);
            U.col(i) = Ua.col(m - 1 - i);
            }
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
        piP = P_sparse.transpose() * pi;
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
    SparseMatrix S = s.asDiagonal() * P_sparse * sinv.asDiagonal();
    SparseMatrix St = SparseMatrix(S.transpose());
    S = 0.5 * (S + St);
    S.makeCompressed();
    return S;
}

double MarkovChainAnalyzer::transitionProbability(Eigen::Index i, Eigen::Index j) const {

    return isSparse ? P_sparse.coeff(i, j) : P_dense(i, j);
}

bool MarkovChainAnalyzer::verifyStationary(double tol) const {

    Vector piP;
    if (isSparse)
        piP = P_sparse.transpose() * pi;
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
    // file schema matches earlier output.
    DenseMatrix S = symmetrizedDense();
    Eigen::SelfAdjointEigenSolver<DenseMatrix> es(S, writeEigenvectors ? Eigen::ComputeEigenvectors : Eigen::EigenvaluesOnly);
    if (es.info() != Eigen::Success)
        throw std::runtime_error("Symmetric eigensystem calculation failed for: " + filePrefix);

    // Descending algebraic order (lambda_1 = 1 first).
    Eigen::VectorXd evalsAsc = es.eigenvalues();
    Eigen::Index N = evalsAsc.size();
    Eigen::VectorXd evals(N);
    for (Eigen::Index k = 0; k < N; ++k)
        evals(k) = evalsAsc(N - 1 - k);

    DenseMatrix P;
    DenseMatrix V;          // pi-orthonormal right eigenvectors of P, columns aligned with evals
    Vector sinv = pi.cwiseSqrt().cwiseInverse();
    if (writeEigenvectors)
        {
        P = denseTransitionMatrix();
        DenseMatrix U = es.eigenvectors();
        V.resize(U.rows(), N);
        for (Eigen::Index k = 0; k < N; ++k)
            V.col(k) = sinv.cwiseProduct(U.col(N - 1 - k));
        }

    std::ofstream evalOut(filePrefix + ".eigenvalues.tsv");
    if (!evalOut)
        throw std::runtime_error("Could not open eigenvalue file: " + filePrefix + ".eigenvalues.tsv");

    evalOut << std::setprecision(17);
    evalOut << "eigen_index\treal\timag\tmodulus\tresidual\n";
    for (Eigen::Index k = 0; k < N; ++k)
        {
        double residual = std::numeric_limits<double>::quiet_NaN();
        if (writeEigenvectors)
            {
            Eigen::VectorXd v = V.col(k);
            double denom = std::max(v.norm(), std::numeric_limits<double>::min());
            residual = (P * v - evals(k) * v).norm() / denom;
            }
        evalOut << k << "\t" << evals(k) << "\t" << 0.0
                << "\t" << std::abs(evals(k)) << "\t" << residual << "\n";
        }

    if (!writeEigenvectors)
        return;

    std::ofstream vecOut(filePrefix + ".eigenvectors.tsv");
    if (!vecOut)
        throw std::runtime_error("Could not open eigenvector file: " + filePrefix + ".eigenvectors.tsv");

    vecOut << std::setprecision(17);
    vecOut << "eigen_index\tstate_index\ttree_hash\treal\timag\n";
    for (Eigen::Index k = 0; k < V.cols(); ++k)
        {
        for (Eigen::Index i = 0; i < V.rows(); ++i)
            {
            uint64_t h = (static_cast<size_t>(i) < stateHashes.size()) ? stateHashes[static_cast<size_t>(i)] : 0;
            vecOut << k << "\t" << i << "\t" << h << "\t"
                   << V(i, k) << "\t" << 0.0 << "\n";
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

