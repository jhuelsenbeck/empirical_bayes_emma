#ifndef Mcmc_hpp
#define Mcmc_hpp

#include <map>
#include <vector>
class Alignment;
class LikelihoodCalculator;
class RandomVariable;
class TreeList;
class ThreadPool;
class Tree;
class TreeList;
class TreeSpace;


class Mcmc {

    public:
                                            Mcmc(void) = delete;
                                            Mcmc(RandomVariable* r, ThreadPool* tp, Alignment* a, TreeList* tl, TreeSpace* ts);
                                           ~Mcmc(void);
        void                                run(std::map<uint64_t,std::pair<double,double>>& treeProbabilities);
        void                                run(std::map<uint64_t,std::pair<double,double>>& treeProbabilities, int numChains, double temperature);
    
    private:
        void                                calculateMaximumLikelihoods(TreeList& treeList, uint64_t currentTree, std::vector<uint64_t>& neighbors, std::vector<std::pair<uint64_t, double>>& neighborhoodInfo);
        std::pair<int,int>                  chooseChains(int numChains);
        double                              chooseTree(std::vector<std::pair<uint64_t, double>>& neighborhoodInfo, uint64_t& tree);
        double                              findTreeProbability(std::vector<std::pair<uint64_t, double>>& neighborhoodInfo, uint64_t& tree);
        LikelihoodCalculator*               getCalculator(void);
        double                              heat(int i, double temperature);
        void                                normalize(double power, std::vector<std::pair<uint64_t, double>>& neighborhoodInfo);
        void                                print(std::map<uint64_t,std::pair<double,double>>& treeProbabilities);
        void                                printToScreen(int n, double curLnL, double newLnL, size_t treeListSize);
        void                                returnCalculator(LikelihoodCalculator* calculator);
        Alignment*                          alignment;
        RandomVariable*                     rng;
        ThreadPool*                         threadPool;
        TreeList*                           treeList;
        TreeSpace*                          treeSpace;
        int                                 chainLength;
        int                                 printFrequency;
        int                                 sampleFrequency;
        int                                 burn;
        std::vector<LikelihoodCalculator*>  activeCalculators;
        std::vector<LikelihoodCalculator*>  calculatorPool;
        std::vector<LikelihoodCalculator*>  allocatedCalculators;
};

#endif
