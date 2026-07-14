#ifndef MarkovChainAnalyzer_hpp
#define MarkovChainAnalyzer_hpp

#include <cassert>
#include <functional>
#include <complex>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseLU>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

class ThreadPool;
class TreeCache;

struct IrreducibilityInfo {

    bool                    irreducible = false;
    double                  threshold = 0.0;
    size_t                  states_reachable_from_0 = 0;
    size_t                  states_that_can_reach_0 = 0;
    size_t                  num_states = 0;
    size_t                  min_out_degree = 0;
    size_t                  max_out_degree = 0;
    double                  min_leave_probability = std::numeric_limits<double>::quiet_NaN();
    size_t                  num_states_with_zero_out_degree = 0;
    size_t                  num_states_with_tiny_leave_probability = 0;
};

struct ThresholdedIrreducibilityInfo {

    std::vector<double>     thresholds;
    std::vector<IrreducibilityInfo> results;
    double                  largest_threshold_irreducible = std::numeric_limits<double>::quiet_NaN();
    bool                    irreducible_at_zero = false;
};

struct DetailedBalanceInfo {

    bool                    reversible = false;
    bool                    skipped = false;
    double                  max_abs_error = 0.0;
    double                  sum_abs_error = 0.0;
    double                  max_relative_error = 0.0;
};

struct TransitionProbabilityInfo {

    size_t                  num_positive_offdiag_transitions = 0;
    double                  min_positive_offdiag_transition = std::numeric_limits<double>::quiet_NaN();
    double                  max_offdiag_transition = std::numeric_limits<double>::quiet_NaN();
    double                  mean_positive_offdiag_transition = std::numeric_limits<double>::quiet_NaN();
    double                  min_leave_probability = std::numeric_limits<double>::quiet_NaN();
    double                  max_leave_probability = std::numeric_limits<double>::quiet_NaN();
    double                  mean_leave_probability = std::numeric_limits<double>::quiet_NaN();
    size_t                  num_states_with_zero_leave_probability = 0;
    std::vector<double>     thresholds;
    std::vector<size_t>     num_transitions_le_threshold;
    std::vector<size_t>     num_states_leave_le_threshold;
};

struct SpectralInfo {

    double                  spectral_gap = std::numeric_limits<double>::quiet_NaN();
    double                  lambda2_abs = std::numeric_limits<double>::quiet_NaN();
    double                  relaxation_time = std::numeric_limits<double>::quiet_NaN();
    double                  worst_case_iact = std::numeric_limits<double>::quiet_NaN();

                            // The MH kernel is reversible, so its spectrum is real. These are the
                            // primary outputs; eigenvalues_complex is retained (imag == 0) only for
                            // backward compatibility with older consumers.
    Eigen::VectorXd         eigenvalues_real;
    double                  lambda2_algebraic = std::numeric_limits<double>::quiet_NaN();
    double                  lambda_min = std::numeric_limits<double>::quiet_NaN();

    Eigen::VectorXcd        eigenvalues_complex;
    Eigen::VectorXd         eigenvalue_moduli;
    Eigen::Index            multiplicity_of_1 = 0;
    bool                    has_complex_eigenvalues = false;
    bool                    computed_sparse = false;
    int                     n_converged = 0;

                            // robustness diagnostics for iterative sparse eigensolves
    bool                    spectral_valid = false;
    std::string             spectral_status = "not_computed";
    double                  max_eigen_residual = std::numeric_limits<double>::quiet_NaN();
    double                  lambda1_error = std::numeric_limits<double>::quiet_NaN();
    int                     ncv_used = 0;
    int                     max_iterations_used = 0;
    double                  tolerance_used = std::numeric_limits<double>::quiet_NaN();
    int                     num_solver_attempts = 0;
};

struct EfficiencyInfo {

                            // The efficiency, or variance ratio, of the chain for estimating the posterior
                            // mean of some function h of the tree. When h is the indicator of a feature --
                            // a particular topology, or the set of topologies carrying a given split --
                            // posterior_mean is that feature's posterior probability, and efficiency is the
                            // factor by which t iterations of the chain are worth fewer than t independent
                            // draws from the posterior when estimating it.
    double                  posterior_mean = std::numeric_limits<double>::quiet_NaN();
    double                  independent_variance = std::numeric_limits<double>::quiet_NaN();
    double                  asymptotic_variance = std::numeric_limits<double>::quiet_NaN();
    double                  efficiency = std::numeric_limits<double>::quiet_NaN();
    double                  integrated_autocorrelation_time = std::numeric_limits<double>::quiet_NaN();
    int                     iterations = 0;
    double                  relative_residual = std::numeric_limits<double>::quiet_NaN();
    bool                    converged = false;
};

struct PeskunComparison {

    bool                    dominates = false;          // this kernel >= other on every shared off-diagonal entry
    size_t                  shared_states = 0;
    size_t                  compared_offdiag = 0;
    size_t                  violations = 0;
    double                  max_violation = 0.0;        // largest amount by which other exceeded this
    double                  min_slack = std::numeric_limits<double>::infinity();
};



class MarkovChainAnalyzer {

    public:
        using DenseMatrix     = Eigen::MatrixXd;
        using Vector          = Eigen::VectorXd;
        using SparseMatrix    = Eigen::SparseMatrix<double>;
        using RowSparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;

        explicit                        MarkovChainAnalyzer(ThreadPool* tp, TreeCache* cache, std::string nme, bool useSparse = true);
                                        MarkovChainAnalyzer(ThreadPool* tp, const SparseMatrix& P, const Vector& pi);
                                        MarkovChainAnalyzer(ThreadPool* tp, const DenseMatrix& P, const Vector& pi);
        double                          stationaryDiscrepancy(void) const;
        bool                            verifyStationary(double tol = 1e-8) const;
        bool                            checkDetailedBalance(double tol = 1e-8) const;
        DetailedBalanceInfo             computeDetailedBalanceInfo(double tol = 1e-8) const;
        IrreducibilityInfo              computeIrreducibilityInfo(double threshold = 0.0, double tinyLeaveProb = 1e-14) const;
        bool                            checkIrreducible(double threshold = 0.0, std::ostream& os = std::cout) const;
        static std::vector<double>      defaultIrreducibilityThresholds(void);
        ThresholdedIrreducibilityInfo   computeThresholdedIrreducibilityInfo(const std::vector<double>& thresholds = defaultIrreducibilityThresholds(), double tinyLeaveProb = 1e-14) const;
        TransitionProbabilityInfo       computeTransitionProbabilityInfo(const std::vector<double>& thresholds = defaultIrreducibilityThresholds()) const;
        SpectralInfo                    computeSpectralInfo(void) const;
        SpectralInfo                    computeSpectralInfoSparse(int nev = 4) const;
        SpectralInfo                    getSpectralInfo(int nev = 4) const;
        void                            clearSpectralCache(void) const;
        DenseMatrix                     computeHittingTimes(void) const;
        Eigen::Index                    getMAPTreeIndex(void) const;

                                        // Mean first-passage times. The expected number of iterations the chain
                                        // needs to visit a given tree for the first time, from every other tree.
                                        // A single sparse solve returns the passage times to one target from all
                                        // B(N) topologies; the full matrix of pairwise passage times is not formed,
                                        // and could not be.
                                        // The target tree is always supplied by the caller, as a hash. The MapTree
                                        // class is the program's authority on which tree is the MAP tree and which
                                        // splits it contains, and passing its hash keeps that the only definition.
        Vector                          meanFirstPassageTimesToState(Eigen::Index target, int maxIterations = 20000, double tolerance = 1.0e-10) const;
        Vector                          meanFirstPassageTimesToTree(uint64_t treeHash, int maxIterations = 20000, double tolerance = 1.0e-10) const;
        void                            writeMeanFirstPassageTimesToTree(uint64_t treeHash, const std::string& fileName, int maxIterations = 20000, double tolerance = 1.0e-10) const;
        Eigen::Index                    stateIndexForHash(uint64_t treeHash) const;

                                        // Efficiency of the chain for estimating the posterior probability of a
                                        // feature of the tree. The argument is an INDICATOR: one for every topology
                                        // carrying the feature and zero for all the rest, so that its posterior mean
                                        // is the feature's posterior probability. The calculation is exact, and needs
                                        // no eigenvalue: it is a single sparse solve of Poisson's equation, and is
                                        // therefore available for the largest state spaces considered here.
        EfficiencyInfo                  efficiencyFor(const Vector& h, int maxIterations = 20000, double tolerance = 1.0e-10) const;
        Vector                          indicatorForTree(uint64_t treeHash) const;
        Vector                          indicatorForTrees(const std::vector<uint64_t>& treeHashes) const;
        static void                     writeEfficiencyTsvHeader(std::ostream& os);
        void                            writeEfficiencyTsvRow(std::ostream& os, const std::string& moveType, double power, const std::string& functional, const EfficiencyInfo& info) const;

        double                          meanHittingTimeToMAP(void) const;
        double                          meanHittingTimeToState(Eigen::Index target) const;
        double                          meanHittingTimeToSet(const std::vector<Eigen::Index>& targets) const;
        double                          meanHittingTimeToPosteriorMass(double targetMass = 0.95) const;
        Vector                          meanReturnTimes(void) const;
        double                          averageAcceptanceRate(void) const;
        double                          entropyRate(void) const;
        double                          kemenyConstant(void) const;
        double                          approximateKemenyConstant(int nev = 32) const;
        double                          kemenyConstantStochastic(int numProbes = 32, int lanczosSteps = 50, unsigned int seed = 0x5eed1234u) const;

                                        // Functional-specific efficiency. For a reversible chain these are exact
                                        // (closed form in the symmetrized eigenbasis) when the full spectrum is
                                        // available (n <= denseStateLimit); otherwise they are truncated to the
                                        // leading slow modes and should be read as estimates.
        double                          integratedAutocorrelationTime(const Vector& f) const;
        double                          asymptoticVariance(const Vector& f) const;
        double                          stationaryVariance(const Vector& f) const;
        double                          chiSquareDistanceFromStart(Eigen::Index startState, int t) const;
        Vector                          chiSquareDistanceCurve(Eigen::Index startState, const std::vector<int>& times) const;
        Vector                          indicatorMAP(void) const;
        Vector                          indicatorPosteriorMassSet(double targetMass) const;

                                        // Peskun ordering: if this kernel dominates other off-diagonally, this
                                        // kernel has provably smaller asymptotic variance for every functional.
                                        // States are matched by tree hash, so the two analyzers need not share
                                        // an indexing (they must share the underlying tree set).
        PeskunComparison                peskunComparison(const MarkovChainAnalyzer& other, double tol = 1e-12) const;

        double                          mixingTimeUpperBound(double epsilon = 1e-6) const;
        double                          mixingTimeUpperBoundValue(double epsilon = 1e-6) const;

                                        // conductance is NP-hard to optimize exactly. These are sweep-cut estimates.
        double                          conductanceForSet(const std::vector<Eigen::Index>& set) const;
        double                          posteriorSweepConductance(void) const;
        double                          eigenvectorSweepConductance(size_t maxDenseStates = 11000) const;
        static void                     writeTsvHeader(std::ostream& os);
        void                            writeTsvRow(std::ostream& os, const std::string& moveType, double power, double epsilon = 1e-6) const;
        void                            printExtendedReport(std::ostream& os = std::cout) const;
        void                            printReport(std::ostream& os = std::cout) const;
        const SparseMatrix&             getTransitionMatrixSparse(void) const { return P_sparse; }
        const DenseMatrix&              getTransitionMatrixDense(void) const { return P_dense; }
        const Vector&                   getPosterior(void) const { return pi; }
        size_t                          numStates(void) const { return n; }

                                        // Exact/small-state output utilities. Intended for <= 8-taxon analyses
                                        // where n = 10395 or smaller. The transition kernel is written in
                                        // coordinate format by default because a dense TSV is very large.
        void                            writeSmallStateAnalysisFiles(const std::string& filePrefix, bool writeDenseKernel = false, bool writeFullEigenvectors = true, bool writeAllPairsHittingTimes = false) const;
        void                            writeTransitionKernelTsv(const std::string& fileName, bool denseFormat = false) const;
        void                            writePosteriorTsv(const std::string& fileName) const;
        void                            writeFullEigenSystemTsv(const std::string& filePrefix, bool writeEigenvectors = true) const;
        void                            writeSmallHittingTimeFiles(const std::string& filePrefix, bool writeAllPairs = false) const;
        const std::vector<uint64_t>&    getStateHashes(void) const { return stateHashes; }

                                        // tuning knobs for million-state analyses.
        void                            setDenseStateLimit(size_t x) { denseStateLimit = x; }
        void                            setExactHittingStateLimit(size_t x) { exactHittingStateLimit = x; }
        void                            setDetailedBalanceStateLimit(size_t x) { detailedBalanceStateLimit = x; }
        void                            setDefaultSparseEigenvalues(int x) { defaultSparseEigenvalues = x; clearSpectralCache(); }

    private:
        void                            initCommon(void);
        void                            applySymmetrizedLaplacian(const Vector& piSqrt, const Vector& y, Vector& out) const;
        void                            sparseMatVec(const Vector& x, Vector& y) const;
        Vector                          posteriorSqrt(void) const;
        void                            buildFromCache(TreeCache* cache);
        bool                            conjugateGradient(const std::function<void(const Vector&, Vector&)>& applyA,
                                                          const Vector& diagonal, const Vector& b, Vector& x,
                                                          double tolerance, int maxIterations,
                                                          int& iterations, double& relativeResidual) const;
        void                            ensureRowSparse(void) const;
        double                          transitionProbability(Eigen::Index i, Eigen::Index j) const;
        std::vector<Eigen::Index>       posteriorMassSet(double targetMass) const;
        double                          conductanceFromOrderingFast(const std::vector<Eigen::Index>& order) const;
        double                          meanHittingTimeToStateQuiet(Eigen::Index target) const;
        double                          meanHittingTimeToSetQuiet(const std::vector<Eigen::Index>& targets) const;
        size_t                          countReachableForward(Eigen::Index start, double threshold) const;
        size_t                          countReachableReverse(Eigen::Index start, double threshold) const;
        static void                     finalizeSpectralInfo(SpectralInfo& info);
        static void                     finalizeSpectralReal(SpectralInfo& info);
        SparseMatrix                    symmetrizedSparse(void) const;
        DenseMatrix                     symmetrizedDense(void) const;
        bool                            fullSymmetricEigensystem(Eigen::VectorXd& evals, DenseMatrix& U) const;
        bool                            leadingSymmetricEigensystem(int kLargest, int kSmallest, Eigen::VectorXd& evals, DenseMatrix& U) const;
        bool                            spectralModesForFunctionals(Eigen::VectorXd& evals, DenseMatrix& U) const;
        DenseMatrix                     denseTransitionMatrix(void) const;
        Vector                          hittingTimesToStateVector(Eigen::Index target) const;
        Vector                          hittingTimesToSetVector(const std::vector<Eigen::Index>& targets) const;
        ThreadPool*                     threadPool;
        SparseMatrix                    P_sparse;
        mutable RowSparseMatrix         P_row_sparse;
        mutable bool                    rowSparseReady = false;
        DenseMatrix                     P_dense;
        Vector                          pi;
        std::vector<uint64_t>           stateHashes;
        std::unordered_map<uint64_t,Eigen::Index> hashToIndex;
        size_t                          n = 0;
        std::string                     name;
        bool                            isSparse = true;
        size_t                          denseStateLimit = 11000;
        size_t                          exactHittingStateLimit = 20000;
        size_t                          detailedBalanceStateLimit = 200000;
        int                             defaultSparseEigenvalues = 4;
        mutable bool                    warnedZeroPosterior = false;
        mutable bool                    spectralCacheValid = false;
        mutable int                     spectralCacheNev = 0;
        mutable SpectralInfo            spectralCache;
};

#endif
