#ifndef Mcmc_hpp
#define Mcmc_hpp

#include <vector>
class Alignment;
class LikelihoodCalculator;
class RandomVariable;
class TreeList;
class ThreadPool;
class Tree;


class Mcmc {

    public:
                                            Mcmc(void) = delete;
                                            Mcmc(RandomVariable* r, ThreadPool* tp, Alignment* a);
                                           ~Mcmc(void);
        void                                run(void);
    
    private:
        void                                calculateMaximumLikelihoods(TreeList& treeList, uint64_t currentTree, std::vector<uint64_t>& neighbors, std::vector<std::pair<uint64_t, double>>& neighborhoodInfo);
        double                              chooseTree(std::vector<std::pair<uint64_t, double>>& neighborhoodInfo, uint64_t& tree);
        double                              findTreeProbability(std::vector<std::pair<uint64_t, double>>& neighborhoodInfo, uint64_t& tree);
        LikelihoodCalculator*               getCalculator(void);
        void                                normalize(double power, std::vector<std::pair<uint64_t, double>>& neighborhoodInfo);
        void                                returnCalculator(LikelihoodCalculator* calculator);
        Alignment*                          alignment;
        RandomVariable*                     rng;
        ThreadPool*                         threadPool;
        int                                 chainLength;
        int                                 printFrequency;
        int                                 sampleFrequency;
        std::vector<LikelihoodCalculator*>  activeCalculators;
        std::vector<LikelihoodCalculator*>  calculatorPool;
        std::vector<LikelihoodCalculator*>  allocatedCalculators;
};

#endif
