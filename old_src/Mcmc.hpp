#ifndef Mcmc_hpp
#define Mcmc_hpp

#include <map>
#include <vector>
class Alignment;
class LikelihoodCalculator;
class NeighborValues;
class RandomVariable;
class TreeList;
class ThreadPool;
class Tree;
class TreeList;
class TreeNeighborhood;
class TreeSamples;
class TreeSpace;


class Mcmc {

    public:
                                            Mcmc(void) = delete;
                                            Mcmc(RandomVariable* r, ThreadPool* tp, Alignment* a, TreeList* tl, TreeSpace* ts);
                                           ~Mcmc(void);
        TreeSamples*                        getSamples(void) { return samples; }
        void                                run(TreeNeighborhood* neighborhood, double power);
        void                                run(TreeNeighborhood* neighborhood, double power, int numChains, double temperature);
    
    private:
        double                              calculateMaximumLikelihood(NeighborValues& vals);
        std::pair<int,int>                  chooseChains(int numChains);
        double                              chooseTree(std::vector<std::pair<uint64_t, double>>& neighborhoodInfo, uint64_t& tree);
        int                                 coldChainIndex(std::vector<int>& chainIndices);
        double                              findTreeProbability(std::vector<std::pair<uint64_t, double>>& neighborhoodInfo, uint64_t& tree);
        LikelihoodCalculator*               getCalculator(void);
        double                              heat(int i, double temperature);
        void                                normalize(double power, NeighborValues& neighbors);
        void                                printToScreen(int n, double curLnL, double newLnL, size_t treeListSize);
        void                                returnCalculator(LikelihoodCalculator* calculator);
        Alignment*                          alignment;
        RandomVariable*                     rng;
        ThreadPool*                         threadPool;
        TreeList*                           treeList;
        TreeSpace*                          treeSpace;
        TreeSamples*                        samples;
        int                                 chainLength;
        int                                 printFrequency;
        int                                 sampleFrequency;
        int                                 burn;
        std::vector<LikelihoodCalculator*>  activeCalculators;
        std::vector<LikelihoodCalculator*>  calculatorPool;
        std::vector<LikelihoodCalculator*>  allocatedCalculators;
};

#endif
