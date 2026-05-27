#ifndef Mcmc_hpp
#define Mcmc_hpp

#include <fstream>
#include <vector>
#include "TreeCache.hpp"
class Alignment;
class LikelihoodCalculator;
class RandomVariable;
class ThreadPool;
class Tree;
class TreeLikelihoods;
class TreeNeighbors;
class TreePartitions;
class TreeSamples;
typedef std::vector<LikelihoodCalculator*> CalculatorVector;



class Mcmc {

    public:
                                        Mcmc(void) = delete;
                                        Mcmc(RandomVariable* r, ThreadPool* p, TreeCache* tc, TreeLikelihoods* tl, TreeNeighbors* tn, Alignment* a);
                                       ~Mcmc(void);
        void                            run(double power);
        void                            run(double power, int numRuns);
        void                            run(double power, int numRuns, int numChains);
    
    private:
        double                          calculateMaximumLikelihood(Tree* currentTree);
        double                          calculateMaximumLikelihood(std::vector<TreeInfo*>& vals);
        std::pair<int,int>              chooseChains(int numChains);
        double                          chooseTree(std::vector<TreeInfo*>& neighbors, std::vector<double>& probs, Tree*& tree, double& lnL);
        int                             coldChainIndex(std::vector<int>& chainIndices);
        double                          findTreeProbability(std::vector<TreeInfo*>& neighbors, std::vector<double>& probs, uint64_t tree);
        LikelihoodCalculator*           getCalculator(void);
        double                          heat(int i, double temperature);
        void                            normalize(double power, std::vector<TreeInfo*>& neighbors, std::vector<double>& probs);
        void                            printTreeToFile(int n, Tree* currentTree);
        void                            printToScreen(int n, double curLnL, double newLnL);
        void                            printToScreen(int n, std::vector<double>& curLnL);
        void                            printToScreen(int n, std::vector<std::vector<double>>& curLnL, std::vector<std::vector<int>>& indices);
        void                            returnCalculator(LikelihoodCalculator* calculator);
        void                            openConvergenceLog(void);
        void                            openTreeFile(void);
        void                            writeConvergenceLine(int cycle);
        std::ofstream                   convergenceLog;
        std::ofstream                   treeStrm;
        RandomVariable*                 rng;
        ThreadPool*                     threadPool;
        Alignment*                      alignment;
        TreeCache*                      treeCache;
        TreeLikelihoods*                treeLikelihoods;
        TreeNeighbors*                  treeNeighbors;
        std::vector<TreePartitions*>    partitions;
        std::vector<TreeSamples*>       samples;
        double                          temperature;
        int                             numChains;
        int                             numCycles;
        int                             printFrequency;
        int                             sampleFrequency;
        CalculatorVector                activeCalculators;
        CalculatorVector                calculatorPool;
        CalculatorVector                allocatedCalculators;
};

#endif
