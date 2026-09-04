#ifndef LikelihoodCalculator_hpp
#define LikelihoodCalculator_hpp

#include <cstdint>
#include <vector>
#include "Threads.hpp"
#include "TransitionMatrix.hpp"
class Alignment;
class ConditionalLikelihoods;
class Node;
class Tree;
struct TreeInfo;


class LikelihoodCalculator : public LikelihoodTask {

    public:
                                    LikelihoodCalculator(void) = delete;
                                    LikelihoodCalculator(Alignment* aln);
                                   ~LikelihoodCalculator(void);
        size_t                      getOffset(void) { return offset; }
        Tree*                       getTree(void) { return tree; }
        TreeInfo*                   getTreeInfo(void) { return treeInfo; }
        double                      lnLikelihood(void);
        double                      lnMarginalLikelihood(void);
        void                        setOffset(size_t x) { offset = x; }
        void                        setTree(Tree* t);
        void                        setTreeInfo(TreeInfo* ti) { treeInfo = ti; }

                                    // Marginal-likelihood switch. When on, lnLikelihood() also integrates the
                                    // branch lengths out under an IID Exp(exponentialPriorRate) prior by a
                                    // Laplace approximation and stores the result on the tree's TreeInfo. Set
                                    // once before the parallel likelihood pass; it is only read there.
        static void                 setComputeMarginalLikelihood(bool x) { computeMarginal = x; }
        static bool                 getComputeMarginalLikelihood(void) { return computeMarginal; }
        static void                 setExponentialPriorRate(double lambda) { exponentialPriorRate = lambda; }
                                    // Hessian used for the Laplace volume term / covariance:
                                    // 0 = diagonal only, 1 = full, by finite differences of the gradient.
        static void                 setMarginalHessianMethod(int m) { hessianMethod = m; }
        static int                  getMarginalHessianMethod(void) { return hessianMethod; }
                                    // Fill the theta-space (log branch length) log-posterior Hessian at the tree's
                                    // current branch lengths, row-major d x d; returns the number of branches d.
                                    // (-H) is the Gaussian precision of the Laplace fit; its inverse is the
                                    // branch-length covariance.
        int                         branchHessianTheta(std::vector<double>& H);
    
    private:
        double                      calculateLFS(double* clsUp, double* clsDn, TransitionMatrix& tMat, TransitionMatrix& fMat, TransitionMatrix& sMat, double& firstDerivative, double& secondDerivative);
        double                      iterateBranch(Node* p);
        double                      mapRefineBranchStep(Node* p, double lambda);
        void                        collectBranchNodes(std::vector<Node*>& order);
        void                        refreshAllConditionalLikelihoods(const std::vector<Node*>& order);
        void                        computeThetaGradient(const std::vector<Node*>& order, double lambda, std::vector<double>& G);
        void                        hessianThetaFiniteDifference(const std::vector<Node*>& order, double lambda, std::vector<double>& H);
        void                        markDnClsAsDirtyFromNode(Node* p);
        void                        markUpClsAsDirtyFromNode(Node* p);
        void                        updateDnConditionalLikelihoods(Node* p);
        void                        updateUpConditionalLikelihoods(Node* p);
        Tree*                       tree;
        Node*                       treeRoot;
        Node*                       treeRootDescendant;
        int*                        patternCount;
        TransitionMatrix*           tiProbs;
        ConditionalLikelihoods**    condLikes;   
        TreeInfo*                   treeInfo;
        static constexpr double     minBrlen = 1.0e-6;
        static constexpr double     maxBrlen = 2.0;
        static bool                 computeMarginal;
        static double               exponentialPriorRate;
        static int                  hessianMethod;
        size_t                      numSites;
        size_t                      numTaxa;
        size_t                      numNodes;
        size_t                      offset;
        FirstDerivatives            firstDerivative;
        SecondDerivatives           secondDerivative; 
};

#endif
