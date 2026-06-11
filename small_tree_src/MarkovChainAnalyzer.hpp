#ifndef MarkovChainAnalyzer_hpp
#define MarkovChainAnalyzer_hpp

#include <cassert>
#include <complex>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseLU>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>

class TreeCache;

class MarkovChainAnalyzer {

public:
    using DenseMatrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;

    // Keep the primary sparse matrix column-major. Spectra::SparseGenMatProd
    // expects this by default and is fastest/least surprising in this form.
    using SparseMatrix = Eigen::SparseMatrix<double>;

    // Secondary row-major copy used only for diagnostics that iterate over rows.
    using RowSparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;

    explicit            MarkovChainAnalyzer(TreeCache* cache, std::string nme, bool useSparse = true);
                        MarkovChainAnalyzer(const SparseMatrix& P, const Vector& pi);
                        MarkovChainAnalyzer(const DenseMatrix& P, const Vector& pi);

    double              stationaryDiscrepancy(void) const;
    bool                verifyStationary(double tol = 1e-8) const;
    bool                checkDetailedBalance(double tol = 1e-8) const;

    struct DetailedBalanceInfo {
        bool            reversible = false;
        bool            skipped = false;
        double          max_abs_error = 0.0;
        double          sum_abs_error = 0.0;
        double          max_relative_error = 0.0;
    };
    DetailedBalanceInfo computeDetailedBalanceInfo(double tol = 1e-8) const;

    struct SpectralInfo {
        double          spectral_gap = std::numeric_limits<double>::quiet_NaN();
        double          lambda2_abs = std::numeric_limits<double>::quiet_NaN();
        double          relaxation_time = std::numeric_limits<double>::quiet_NaN();
        double          worst_case_iact = std::numeric_limits<double>::quiet_NaN();
        Eigen::VectorXcd eigenvalues_complex;
        Vector          eigenvalue_moduli;
        Eigen::Index    multiplicity_of_1 = 0;
        bool            has_complex_eigenvalues = false;
        bool            computed_sparse = false;
        int             n_converged = 0;
    };
    
    SpectralInfo        computeSpectralInfo(void) const;
    SpectralInfo        computeSpectralInfoSparse(int nev = 4) const;
    SpectralInfo        getSpectralInfo(int nev = 4) const;
    void                clearSpectralCache(void) const;

    DenseMatrix         computeHittingTimes(void) const;
    Eigen::Index        getMAPTreeIndex(void) const;
    double              meanHittingTimeToMAP(void) const;
    double              meanHittingTimeToState(Eigen::Index target) const;
    double              meanHittingTimeToSet(const std::vector<Eigen::Index>& targets) const;
    double              meanHittingTimeToPosteriorMass(double targetMass = 0.95) const;
    Vector              meanReturnTimes(void) const;

    double              averageAcceptanceRate(void) const;
    double              entropyRate(void) const;
    double              kemenyConstant(void) const;
    double              approximateKemenyConstant(int nev = 32) const;
    double              mixingTimeUpperBound(double epsilon = 1e-6) const;
    double              mixingTimeUpperBoundValue(double epsilon = 1e-6) const;

                        // conductance is NP-hard to optimize exactly. These are sweep-cut estimates.
    double              conductanceForSet(const std::vector<Eigen::Index>& set) const;
    double              posteriorSweepConductance(void) const;
    double              eigenvectorSweepConductance(size_t maxDenseStates = 10000) const;

    static void         writeTsvHeader(std::ostream& os);
    void                writeTsvRow(std::ostream& os, const std::string& moveType, double power, double epsilon = 1e-6) const;

    void                printExtendedReport(std::ostream& os = std::cout) const;
    void                printReport(std::ostream& os = std::cout) const;

    const SparseMatrix& getTransitionMatrixSparse(void) const { return P_sparse; }
    const DenseMatrix&  getTransitionMatrixDense(void) const { return P_dense; }
    const Vector&       getPosterior(void) const { return pi; }
    size_t              numStates(void) const { return n; }

                        // tuning knobs for million-state analyses.
    void                setDenseStateLimit(size_t x) { denseStateLimit = x; }
    void                setExactHittingStateLimit(size_t x) { exactHittingStateLimit = x; }
    void                setDetailedBalanceStateLimit(size_t x) { detailedBalanceStateLimit = x; }
    void                setDefaultSparseEigenvalues(int x) { defaultSparseEigenvalues = x; clearSpectralCache(); }

private:
    void                initCommon(void);
    void                buildFromCache(TreeCache* cache);
    void                ensureRowSparse(void) const;
    double              transitionProbability(Eigen::Index i, Eigen::Index j) const;
    std::vector<Eigen::Index> posteriorMassSet(double targetMass) const;
    double              conductanceFromOrderingFast(const std::vector<Eigen::Index>& order) const;
    double              meanHittingTimeToStateQuiet(Eigen::Index target) const;
    double              meanHittingTimeToSetQuiet(const std::vector<Eigen::Index>& targets) const;
    static void         finalizeSpectralInfo(SpectralInfo& info);

    SparseMatrix        P_sparse;
    mutable RowSparseMatrix P_row_sparse;
    mutable bool        rowSparseReady = false;
    DenseMatrix         P_dense;
    Vector              pi;
    size_t              n = 0;
    std::string         name;
    bool                isSparse = true;

    size_t              denseStateLimit = 10000;
    size_t              exactHittingStateLimit = 20000;
    size_t              detailedBalanceStateLimit = 200000;
    int                 defaultSparseEigenvalues = 4;

    mutable bool        spectralCacheValid = false;
    mutable int         spectralCacheNev = 0;
    mutable SpectralInfo spectralCache;
};

#endif
