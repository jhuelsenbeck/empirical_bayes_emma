#include <Eigen/SparseLU>
#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <limits>
#include <numeric>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include "MarkovChainAnalyzer.hpp"
#include "TreeCache.hpp"



namespace {

double quietNaNValue(void) {

    return std::numeric_limits<double>::quiet_NaN();
}

bool finitePositive(double x) {

    return std::isfinite(x) && x > 0.0;
}

}

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
    stateHashes.resize(static_cast<size_t>(n), 0);
    initCommon();
}

MarkovChainAnalyzer::MarkovChainAnalyzer(const DenseMatrix& P, const Vector& pi)
    : P_dense(P), pi(pi), n(P.rows()), isSparse(false) {
    
    stateHashes.resize(static_cast<size_t>(n), 0);
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
    stateHashes.clear();
    stateHashes.resize(n, 0);
    Eigen::Index idx = 0;
    for (const auto& [h, info] : tcache) 
        {
        if (info)
            {
            hashToIdx[h] = idx;
            stateHashes[static_cast<size_t>(idx)] = h;
            ++idx;
            }
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


std::vector<double> MarkovChainAnalyzer::defaultIrreducibilityThresholds(void) {

    return {0.0, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6};
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

SpectralInfo MarkovChainAnalyzer::computeSpectralInfo(void) const {

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
    info.lambda1_error = (info.eigenvalues_complex.size() > 0)
                       ? std::abs(info.eigenvalues_complex(0) - std::complex<double>(1.0, 0.0))
                       : std::numeric_limits<double>::quiet_NaN();
    info.spectral_valid = (info.eigenvalues_complex.size() >= 2 &&
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

    nev = std::max(2, nev);
    nev = std::min(nev, static_cast<int>(n) - 1);

    // A few increasingly conservative ARPACK/Lanczos-style attempts.
    // Larger ncv costs more memory, but is often much more stable when many
    // eigenvalues are clustered near 1, which is exactly the near-reducible case.
    struct SolverAttempt {
        int ncv;
        int maxit;
        double tol;
    };

    std::vector<SolverAttempt> attempts;
    attempts.push_back({std::min(static_cast<int>(n), std::max(2 * nev + 1, 20)), 1000, 1e-6});
    attempts.push_back({std::min(static_cast<int>(n), std::max(4 * nev + 20, 80)), 3000, 1e-7});
    attempts.push_back({std::min(static_cast<int>(n), std::max(6 * nev + 40, 160)), 6000, 1e-8});

    using Op = Spectra::SparseGenMatProd<double>; // column-major SparseMatrix
    Op op(P_sparse);

    double bestResidual = std::numeric_limits<double>::infinity();
    int attemptNumber = 0;

    for (const SolverAttempt& attempt : attempts) 
        {
        attemptNumber++;

        Spectra::GenEigsSolver<Op> eigs(op, nev, attempt.ncv);
        eigs.init();
        int nconv = static_cast<int>(eigs.compute(Spectra::SortRule::LargestMagn, attempt.maxit, attempt.tol));

        SpectralInfo info;
        info.computed_sparse = true;
        info.n_converged = nconv;
        info.ncv_used = attempt.ncv;
        info.max_iterations_used = attempt.maxit;
        info.tolerance_used = attempt.tol;
        info.num_solver_attempts = attemptNumber;
        info.spectral_status = "insufficient_converged_eigenvalues";

        if (nconv < 2) 
            {
            if (nconv > bestInfo.n_converged)
                bestInfo = info;
            continue;
            }

        Eigen::VectorXcd evals_complex = eigs.eigenvalues();
        Eigen::MatrixXcd evecs_complex = eigs.eigenvectors();

        struct EigenPairRecord {
            double modulus;
            std::complex<double> value;
            Eigen::Index original_index;
        };

        std::vector<EigenPairRecord> ev;
        ev.reserve(static_cast<size_t>(evals_complex.size()));
        for (Eigen::Index i = 0; i < evals_complex.size(); i++)
            ev.push_back({std::abs(evals_complex(i)), evals_complex(i), i});
        std::sort(ev.begin(), ev.end(), [](const auto& a, const auto& b) { return a.modulus > b.modulus; });

        Eigen::Index m = static_cast<Eigen::Index>(ev.size());
        info.eigenvalues_complex.resize(m);
        info.eigenvalue_moduli.resize(m);
        for (Eigen::Index i = 0; i < m; i++) 
            {
            info.eigenvalue_moduli(i) = ev[static_cast<size_t>(i)].modulus;
            info.eigenvalues_complex(i) = ev[static_cast<size_t>(i)].value;
            if (std::abs(ev[static_cast<size_t>(i)].value.imag()) > 1e-10)
                info.has_complex_eigenvalues = true;
            if (std::abs(ev[static_cast<size_t>(i)].value - std::complex<double>(1.0, 0.0)) < 1e-8)
                ++info.multiplicity_of_1;
            }

        finalizeSpectralInfo(info);

        // Residual check: ||P v - lambda v|| / ||v|| for the converged eigenpairs.
        // This is the most important guard against silently using bad eigenvalues.
        double maxResidual = 0.0;
        for (Eigen::Index k = 0; k < m; k++) 
            {
            Eigen::Index original = ev[static_cast<size_t>(k)].original_index;
            Eigen::VectorXcd v = evecs_complex.col(original);
            Eigen::VectorXcd Pv = P_sparse * v;
            double denom = std::max(v.norm(), std::numeric_limits<double>::min());
            double resid = (Pv - ev[static_cast<size_t>(k)].value * v).norm() / denom;
            maxResidual = std::max(maxResidual, resid);
            }
        info.max_eigen_residual = maxResidual;
        info.lambda1_error = (m > 0) ? std::abs(info.eigenvalues_complex(0) - std::complex<double>(1.0, 0.0))
                                    : std::numeric_limits<double>::quiet_NaN();

        bool enough = (info.n_converged >= 2 && info.eigenvalue_moduli.size() >= 2);
        bool lambda1OK = std::isfinite(info.lambda1_error) && info.lambda1_error < 1e-5;
        bool residualOK = std::isfinite(info.max_eigen_residual) && info.max_eigen_residual < 1e-5;
        bool lambda2OK = std::isfinite(info.lambda2_abs) && info.lambda2_abs <= 1.0 + 1e-7;

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
        for (Eigen::Index i = 0; i < N; i++) 
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

MarkovChainAnalyzer::Vector MarkovChainAnalyzer::meanReturnTimes(void) const {

    return pi.array().inverse();
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

double MarkovChainAnalyzer::kemenyConstant(void) const {

    if (n > denseStateLimit) 
        {
        std::cout << "   * Exact Kemeny's constant skipped for n = " << n
                  << "; use approximateKemenyConstant() for a leading-eigenvalue approximation\n";
        return std::numeric_limits<double>::quiet_NaN();
        }

    SpectralInfo spec = computeSpectralInfo();
    std::complex<double> K(0.0, 0.0);
    for (Eigen::Index i = 0; i < spec.eigenvalues_complex.size(); i++) 
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
    if (!spec.spectral_valid || spec.eigenvalues_complex.size() < 2)
        return std::numeric_limits<double>::quiet_NaN();

    std::complex<double> K(0.0, 0.0);
    for (Eigen::Index i = 0; i < spec.eigenvalues_complex.size(); i++) 
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



MarkovChainAnalyzer::DenseMatrix MarkovChainAnalyzer::denseTransitionMatrix(void) const {

    return isSparse ? DenseMatrix(P_sparse) : P_dense;
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

void MarkovChainAnalyzer::writeFullEigenSystemTsv(const std::string& filePrefix, bool writeEigenvectors) const {

    DenseMatrix P = denseTransitionMatrix();

    // EigenSolver computes right eigenvectors: P v = lambda v.
    Eigen::EigenSolver<DenseMatrix> solver(P, writeEigenvectors);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("Dense eigensystem calculation failed for: " + filePrefix);

    Eigen::VectorXcd evals = solver.eigenvalues();

    std::ofstream evalOut(filePrefix + ".eigenvalues.tsv");
    if (!evalOut)
        throw std::runtime_error("Could not open eigenvalue file: " + filePrefix + ".eigenvalues.tsv");

    evalOut << std::setprecision(17);
    evalOut << "eigen_index\treal\timag\tmodulus\tresidual\n";

    Eigen::MatrixXcd evecs;
    if (writeEigenvectors)
        evecs = solver.eigenvectors();

    for (Eigen::Index k = 0; k < evals.size(); ++k)
        {
        double residual = std::numeric_limits<double>::quiet_NaN();
        if (writeEigenvectors)
            {
            Eigen::VectorXcd v = evecs.col(k);
            Eigen::VectorXcd Pv = P.cast<std::complex<double> >() * v;
            double denom = std::max(v.norm(), std::numeric_limits<double>::min());
            residual = (Pv - evals(k) * v).norm() / denom;
            }
        evalOut << k << "\t" << evals(k).real() << "\t" << evals(k).imag()
                << "\t" << std::abs(evals(k)) << "\t" << residual << "\n";
        }

    if (!writeEigenvectors)
        return;

    std::ofstream vecOut(filePrefix + ".eigenvectors.tsv");
    if (!vecOut)
        throw std::runtime_error("Could not open eigenvector file: " + filePrefix + ".eigenvectors.tsv");

    vecOut << std::setprecision(17);
    vecOut << "eigen_index\tstate_index\ttree_hash\treal\timag\n";
    for (Eigen::Index k = 0; k < evecs.cols(); ++k)
        {
        for (Eigen::Index i = 0; i < evecs.rows(); ++i)
            {
            uint64_t h = (static_cast<size_t>(i) < stateHashes.size()) ? stateHashes[static_cast<size_t>(i)] : 0;
            vecOut << k << "\t" << i << "\t" << h << "\t"
                   << evecs(i, k).real() << "\t" << evecs(i, k).imag() << "\n";
            }
        }
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
