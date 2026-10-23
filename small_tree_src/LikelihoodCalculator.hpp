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
        static void                 setComputeMarginalLikelihood(bool x) { computeMarginal = x; }
        static bool                 getComputeMarginalLikelihood(void) { return computeMarginal; }
        static void                 setExponentialPriorRate(double lambda) { exponentialPriorRate = lambda; }
        static void                 setMarginalHessianMethod(int m) { hessianMethod = m; }
        static int                  getMarginalHessianMethod(void) { return hessianMethod; }
        int                         branchHessianTheta(std::vector<double>& H);

                                    // How the maximum-likelihood branch lengths are found: Joint (the default)
                                    // optimizes the three branches at each interior node together; SingleBranch
                                    // is the earlier one-branch-at-a-time Newton, kept for comparison.
        enum class                  BranchOptimizer { Joint, SingleBranch };
        static void                 setBranchOptimizer(BranchOptimizer b) { branchOptimizer = b; }
        static BranchOptimizer      getBranchOptimizer(void) { return branchOptimizer; }

    private:
        double                      lnLikelihoodSingleBranch(void);
        double                      lnLikelihoodJoint(void);
        void                        jcUp(Node* p);
        void                        jcDn(Node* c);
        double                      rootEdgeLnL(void);
        void                        smoothNode(Node* v);
        void                        optimizeNode(Node* v);
        void                        setNodeCoefficients(Node* v);
        void                        nodeDerivatives(const double* e, double* g, double* h);
        double                      nodeGain(const double* e);
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
        static BranchOptimizer      branchOptimizer;
        std::vector<double>         c0;                 // joint optimizer: the node's polynomial coefficients,
        std::vector<double>         cab;                //   by pattern
        std::vector<double>         cac;
        std::vector<double>         cbc;
        std::vector<double>         cabc;
        std::vector<double>         nodeL;              // pattern likelihoods at the current point
        std::vector<double>         nodeTrialL;         // and at the trial point
        size_t                      numSites;
        size_t                      numTaxa;
        size_t                      numNodes;
        size_t                      offset;
        FirstDerivatives            firstDerivative;
        SecondDerivatives           secondDerivative; 
};

#endif
