#include <Eigen/SparseLU>
#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include "MarkovChainAnalyzer.hpp"
#include "TreeCache.hpp"



MarkovChainAnalyzer::MarkovChainAnalyzer(TreeCache* cache, std::string nme, bool useSparse)
    : isSparse(useSparse), name(nme) {
    
    if (!cache)
        throw std::invalid_argument("Null TreeCache");
    buildFromCache(cache);
    initCommon();
}

MarkovChainAnalyzer::MarkovChainAnalyzer(const SparseMatrix& P, const Vector& pi)
    : P_sparse(P), pi(pi), n(P.rows()), isSparse(true) {
    
    P_sparse.makeCompressed();
    initCommon();
}

MarkovChainAnalyzer::MarkovChainAnalyzer(const DenseMatrix& P, const Vector& pi)
    : P_dense(P), pi(pi), n(P.rows()), isSparse(false) {
    
    initCommon();
}

void MarkovChainAnalyzer::initCommon(void) {

    if (std::abs(pi.sum() - 1.0) > 1e-8)
        std::cerr << "Warning: Posterior does not sum to 1 (sum = " << pi.sum() << ")\n";
}

void MarkovChainAnalyzer::clearSpectralCache(void) const {

    spectralCacheValid = false;
    spectralCacheNev = 0;
    spectralCache = SpectralInfo();
}

void MarkovChainAnalyzer::ensureRowSparse(void) const {

    if (!isSparse || rowSparseReady)
        return;
    P_row_sparse = P_sparse;
    P_row_sparse.makeCompressed();
    rowSparseReady = true;
}

void MarkovChainAnalyzer::buildFromCache(TreeCache* cache) {

    TreeCacheMap& tcache = cache->getCache();
    n = tcache.size();
    if (n == 0)
        throw std::runtime_error("Empty TreeCache");

    std::unordered_map<uint64_t, Eigen::Index> hashToIdx;
    hashToIdx.reserve(n);
    Eigen::Index idx = 0;
    for (const auto& [h, info] : tcache) 
        {
        if (info)
            hashToIdx[h] = idx++;
        }

    pi = Vector::Zero(static_cast<Eigen::Index>(n));
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(n * 20);

    for (const auto& [h_from, info_from] : tcache) 
        {
        if (!info_from) continue;
        Eigen::Index i = hashToIdx[h_from];
        pi(i) = info_from->posteriorProbability;
        double sumTransition = 0.0;

        for (size_t k = 0; k < info_from->neighbors.size(); ++k) 
            {
            TreeInfo* info_to = info_from->neighbors[k];
            if (!info_to) continue;

            auto it = hashToIdx.find(info_to->hash);
            if (it == hashToIdx.end()) continue;
            Eigen::Index j = it->second;

            double q_ij = (k < info_from->neighborProposalProbabilities.size())
                        ? info_from->neighborProposalProbabilities[k] : 0.0;
            if (q_ij <= 0.0) continue;

            double q_ji = 0.0;
            for (size_t m = 0; m < info_to->neighbors.size(); ++m) 
                {
                if (info_to->neighbors[m] && info_to->neighbors[m]->hash == h_from) 
                    {
                    if (m < info_to->neighborProposalProbabilities.size())
                        q_ji = info_to->neighborProposalProbabilities[m];
                    break;
                    }
                }

            double pi_i = info_from->posteriorProbability;
            double pi_j = info_to->posteriorProbability;
            if (pi_i <= 0.0 || q_ji <= 0.0) 
                continue;

            double ratio = (pi_j * q_ji) / (pi_i * q_ij);
            double alpha = std::min(1.0, ratio);
            double trans_prob = q_ij * alpha;

            if (trans_prob > 0.0) 
                {
                triplets.emplace_back(i, j, trans_prob);
                sumTransition += trans_prob;
                }
            }

        double selfProb = 1.0 - sumTransition;
        if (selfProb < 0.0 && selfProb > -1e-12) selfProb = 0.0;
        if (selfProb < 0.0)
            std::cerr << "Warning: negative self transition at row " << i << ": " << selfProb << "\n";
        triplets.emplace_back(i, i, std::max(0.0, selfProb));
        }

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

double MarkovChainAnalyzer::stationaryDiscrepancy(void) const {

    Vector piP;
    if (isSparse)
        piP = P_sparse.transpose() * pi;
    else
        piP = P_dense.transpose() * pi;
    return (piP - pi).norm();
}

bool MarkovChainAnalyzer::verifyStationary(double tol) const {

    Vector piP;
    if (isSparse)
        piP = P_sparse.transpose() * pi;
    else
        piP = P_dense.transpose() * pi;
    double err = (piP - pi).norm();
    std::cout << std::fixed << std::setprecision(20);
    std::cout << "   * Stationary verification discrepancy: " << err << "\n";
    return err < tol;
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

MarkovChainAnalyzer::DetailedBalanceInfo MarkovChainAnalyzer::computeDetailedBalanceInfo(double tol) const {

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

MarkovChainAnalyzer::SpectralInfo MarkovChainAnalyzer::computeSpectralInfo(void) const {

    if (n > denseStateLimit) 
        {
        if (isSparse)
            return computeSpectralInfoSparse(defaultSparseEigenvalues);
        std::cerr << "Skipping dense eigensystem for n = " << n << " because it exceeds denseStateLimit.\n";
        return SpectralInfo();
        }

    DenseMatrix P = isSparse ? DenseMatrix(P_sparse) : P_dense;
    Eigen::EigenSolver<DenseMatrix> solver(P);
    Eigen::VectorXcd evals_complex = solver.eigenvalues();

    std::vector<std::pair<double, std::complex<double>>> ev;
    ev.reserve(static_cast<size_t>(evals_complex.size()));
    for (Eigen::Index i = 0; i < evals_complex.size(); ++i)
        ev.push_back({std::abs(evals_complex(i)), evals_complex(i)});
    std::sort(ev.begin(), ev.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

    SpectralInfo info;
    info.n_converged = static_cast<int>(ev.size());
    info.eigenvalues_complex.resize(static_cast<Eigen::Index>(ev.size()));
    info.eigenvalue_moduli.resize(static_cast<Eigen::Index>(ev.size()));
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(ev.size()); ++i) 
        {
        info.eigenvalue_moduli(i) = ev[static_cast<size_t>(i)].first;
        info.eigenvalues_complex(i) = ev[static_cast<size_t>(i)].second;
        if (std::abs(ev[static_cast<size_t>(i)].second.imag()) > 1e-10)
            info.has_complex_eigenvalues = true;
        if (std::abs(ev[static_cast<size_t>(i)].first - 1.0) < 1e-8)
            ++info.multiplicity_of_1;
        }
    finalizeSpectralInfo(info);
    return info;
}

MarkovChainAnalyzer::SpectralInfo MarkovChainAnalyzer::computeSpectralInfoSparse(int nev) const {

    if (!isSparse || n <= denseStateLimit)
        return computeSpectralInfo();

    nev = std::max(2, nev);
    nev = std::min(nev, static_cast<int>(n) - 1);

    using Op = Spectra::SparseGenMatProd<double>; // column-major SparseMatrix
    Op op(P_sparse);

    int ncv = std::min(static_cast<int>(n), std::max(2 * nev + 1, 20));
    Spectra::GenEigsSolver<Op> eigs(op, nev, ncv);

    eigs.init();
    int nconv = static_cast<int>(eigs.compute(Spectra::SortRule::LargestMagn, 1000, 1e-6));

    SpectralInfo info;
    info.computed_sparse = true;
    info.n_converged = nconv;

    if (nconv >= 2) 
        {
        Eigen::VectorXcd evals_complex = eigs.eigenvalues();
        std::vector<std::pair<double, std::complex<double>>> ev;
        ev.reserve(static_cast<size_t>(evals_complex.size()));
        for (Eigen::Index i = 0; i < evals_complex.size(); ++i)
            ev.push_back({std::abs(evals_complex(i)), evals_complex(i)});
        std::sort(ev.begin(), ev.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

        Eigen::Index m = static_cast<Eigen::Index>(ev.size());
        info.eigenvalues_complex.resize(m);
        info.eigenvalue_moduli.resize(m);
        for (Eigen::Index i = 0; i < m; ++i) 
            {
            info.eigenvalue_moduli(i) = ev[static_cast<size_t>(i)].first;
            info.eigenvalues_complex(i) = ev[static_cast<size_t>(i)].second;
            if (std::abs(ev[static_cast<size_t>(i)].second.imag()) > 1e-10)
                info.has_complex_eigenvalues = true;
            if (std::abs(ev[static_cast<size_t>(i)].first - 1.0) < 1e-8)
                ++info.multiplicity_of_1;
            }
        finalizeSpectralInfo(info);
        } 
    else 
        {
        std::cerr << "Warning: Only " << nconv << " eigenvalues converged.\n";
        }
    return info;
}

MarkovChainAnalyzer::SpectralInfo MarkovChainAnalyzer::getSpectralInfo(int nev) const {

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
    for (Eigen::Index i = 0; i < N; ++i)
        A.row(i) += pi.transpose();

    DenseMatrix Z = A.inverse();
    DenseMatrix M = DenseMatrix::Zero(N, N);
    for (Eigen::Index i = 0; i < N; ++i)
        for (Eigen::Index j = 0; j < N; ++j)
            M(i, j) = (i == j) ? 0.0 : (Z(j, j) - Z(i, j)) / pi(j);
    return M;
}

double MarkovChainAnalyzer::meanHittingTimeToMAP(void) const {

    return meanHittingTimeToState(getMAPTreeIndex());
}

double MarkovChainAnalyzer::meanHittingTimeToState(Eigen::Index target) const {

    if (target < 0 || target >= static_cast<Eigen::Index>(n))
        return std::numeric_limits<double>::quiet_NaN();
    if (n > exactHittingStateLimit) 
        {
        std::cout << "   * Hitting time to state: skipped for n = " << n
                  << " because exact sparse linear solve is too expensive at this size\n";
        return std::numeric_limits<double>::quiet_NaN();
        }

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
        } 
    else 
        {
        DenseMatrix A = DenseMatrix::Identity(N, N) - P_dense;
        A.row(target) = DenseMatrix::Zero(1, N);
        A(target, target) = 1.0;
        Eigen::ColPivHouseholderQR<DenseMatrix> solver(A);
        h = solver.solve(b);
        }

    double avg = h.mean();
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

MarkovChainAnalyzer::Vector MarkovChainAnalyzer::meanReturnTimes(void) const {

    return pi.array().inverse();
}

double MarkovChainAnalyzer::entropyRate(void) const {

    double H = 0.0;
    Eigen::Index N = static_cast<Eigen::Index>(n);
    if (isSparse) 
        {
        ensureRowSparse();
        for (Eigen::Index i = 0; i < N; ++i) 
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
        for (Eigen::Index i = 0; i < N; ++i)
            for (Eigen::Index j = 0; j < N; ++j)
                if (P_dense(i, j) > 0.0)
                    H -= pi(i) * P_dense(i, j) * std::log(P_dense(i, j));
        }
    return H;
}

double MarkovChainAnalyzer::kemenyConstant(void) const {

    if (n > denseStateLimit) 
        {
        std::cout << "   * Exact Kemeny's constant skipped for n = " << n
                  << "; use approximateKemenyConstant() for a leading-eigenvalue approximation\n";
        return std::numeric_limits<double>::quiet_NaN();
        }

    SpectralInfo spec = computeSpectralInfo();
    std::complex<double> K(0.0, 0.0);
    for (Eigen::Index i = 0; i < spec.eigenvalues_complex.size(); ++i) 
        {
        std::complex<double> lambda = spec.eigenvalues_complex(i);
        if (std::abs(lambda - std::complex<double>(1.0, 0.0)) < 1e-8)
            continue;
        K += 1.0 / (1.0 - lambda);
        }
    return K.real();
}

double MarkovChainAnalyzer::approximateKemenyConstant(int nev) const {

    SpectralInfo spec = getSpectralInfo(std::max(nev, defaultSparseEigenvalues));
    std::complex<double> K(0.0, 0.0);
    for (Eigen::Index i = 0; i < spec.eigenvalues_complex.size(); ++i) 
        {
        std::complex<double> lambda = spec.eigenvalues_complex(i);
        if (std::abs(lambda - std::complex<double>(1.0, 0.0)) < 1e-8)
            continue;
        K += 1.0 / (1.0 - lambda);
        }
    return K.real();
}

double MarkovChainAnalyzer::mixingTimeUpperBoundValue(double epsilon) const {

    SpectralInfo spec = getSpectralInfo(defaultSparseEigenvalues);
    double minPi = pi.minCoeff();
    if (minPi <= 0.0)
        minPi = 1.0 / static_cast<double>(n);
    if (!(spec.spectral_gap > 0.0))
        return std::numeric_limits<double>::infinity();
    return (1.0 / spec.spectral_gap) * std::log(1.0 / (epsilon * minPi));
}

double MarkovChainAnalyzer::mixingTimeUpperBound(double epsilon) const {

    SpectralInfo spec = getSpectralInfo(defaultSparseEigenvalues);
    double minPi = pi.minCoeff();
    if (minPi <= 0.0)
        minPi = 1.0 / static_cast<double>(n);
    double bound = (1.0 / spec.spectral_gap) * std::log(1.0 / (epsilon * minPi));
    std::cout << "   * Mixing time upper bound (epsilon=" << epsilon << "): approx. " << bound << " steps\n";
    return bound;
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
    for (Eigen::Index i = 0; i < N; ++i)
        if (inSet[static_cast<size_t>(i)])
            piA += pi(i);
    double denom = std::min(piA, 1.0 - piA);
    if (denom <= 0.0)
        return std::numeric_limits<double>::quiet_NaN();

    double flow = 0.0;
    if (isSparse) 
        {
        ensureRowSparse();
        for (Eigen::Index i = 0; i < N; ++i) 
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

double MarkovChainAnalyzer::posteriorSweepConductance(void) const {

    std::vector<Eigen::Index> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [this](Eigen::Index a, Eigen::Index b) {
        return pi(a) > pi(b);
    });
    return conductanceFromOrderingFast(order);
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

void MarkovChainAnalyzer::writeTsvHeader(std::ostream& os) {

    os << "move_type"
       << "\tpower"
       << "\tanalysis_name"
       << "\tnum_states"
       << "\tstationary_discrepancy"
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

    os << moveType
       << "\t" << power
       << "\t" << name
       << "\t" << n
       << "\t" << statErr
       << "\t" << (dbChecked ? 1 : 0)
       << "\t" << (db.reversible ? 1 : 0)
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
       << "\n";
}

void MarkovChainAnalyzer::printReport(std::ostream& os) const {

    os << "   Analysis for " << name << "\n";
    os << "   * Number of trees: " << n << "\n";

    verifyStationary();
    if (n <= detailedBalanceStateLimit)
        checkDetailedBalance();
    else
        os << "   * Detailed balance: skipped for large n\n";

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

double MarkovChainAnalyzer::transitionProbability(Eigen::Index i, Eigen::Index j) const {

    return isSparse ? P_sparse.coeff(i, j) : P_dense(i, j);
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
