#ifndef LikelihoodCalculator_hpp
#define LikelihoodCalculator_hpp

#include <vector>
#include "Threads.hpp"
#include "TransitionMatrix.hpp"
class Alignment;
class ConditionalLikelihoods;
class Node;
class Tree;
class TreeInfo;



class LikelihoodCalculator : public LikelihoodTask {

    public:
                                    LikelihoodCalculator(void) = delete;
                                    LikelihoodCalculator(Alignment* aln);
                                   ~LikelihoodCalculator(void);
        size_t                      getOffset(void) { return offset; }
        Tree*                       getTree(void) { return tree; }
        double                      lnLikelihood(void);
        void                        setOffset(size_t x) { offset = x; }
        void                        setTree(Tree* t);
    
    private:
        double                      calculateLFS(double* clsUp, double* clsDn, TransitionMatrix& tMat, TransitionMatrix& fMat, TransitionMatrix& sMat, double& firstDerivative, double& secondDerivative);
        double                      iterateBranch(Node* p);
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
        static constexpr double     minBrlen = 1.0e-6;
        static constexpr double     maxBrlen = 2.0;
        size_t                      numSites;
        size_t                      numTaxa;
        size_t                      numNodes;
        size_t                      offset;
        FirstDerivatives            firstDerivative;
        SecondDerivatives           secondDerivative; 
};

#endif
