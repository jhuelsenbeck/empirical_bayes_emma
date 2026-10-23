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
