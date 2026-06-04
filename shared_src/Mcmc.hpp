#ifndef Mcmc_hpp
#define Mcmc_hpp

#include <cstdint>
#include <fstream>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "Threads.hpp"
#include "TreeCache.hpp"

class Alignment;
class LikelihoodCalculator;
class RandomVariable;
class TreeSamples;

class Mcmc {

    public:
                                    Mcmc(RandomVariable* r, TreeCache* tc, Alignment* a, bool tf=false, std::string cfn="");
                                   ~Mcmc(void);
        void                        run(std::string label, double power, int nNeighbors);
        void                        run(std::string label, double power, int nNeighbors, int numRuns);
        void                        run(std::string label, double power, int nNeighbors, int numRuns, int numHeatedChains);

    private:
        std::pair<int,int>          chooseChains(int numChains);
        TreeInfo*                   chooseInitialTreeInfo(void);
        TreeInfo*                   chooseTreeInfo(TreeInfo* currentInfo, double& proposalProbability);
        TreeInfo*                   chooseTreeInfo(TreeInfo* currentInfo, double& proposalProbability, int n);
        int                         coldChainIndex(std::vector<int>& chainIndices);
        void                        deleteSamplesAndPartitions(void);
        double                      findTreeProbability(TreeInfo* fromInfo, uint64_t toHash);
        double                      findTreeProbability(TreeInfo* fromInfo, uint64_t toHash, int n);
        LikelihoodCalculator*       getCalculator(void);
        double                      heat(int i, double temperature);
        void                        openConvergenceLog(size_t numReplicates);
        void                        printToScreen(int n, double curLnL, double newLnL);
        void                        printToScreen(int n, std::vector<double>& curLnL);
        void                        printToScreen(int n, std::vector<std::vector<double>>& curLnL, std::vector<std::vector<int>>& indices);
        void                        returnCalculator(LikelihoodCalculator* calculator);
        bool                        shouldSample(uint32_t cycle);
        void                        writeConvergenceLine(int cycle);

        RandomVariable*             rng;
        Alignment*                  alignment;
        TreeCache*                  treeCache;
        bool                        expandedOutput;
        std::string                 convergenceLogFileName;
        int                         numChains;
        double                      temperature;
        int                         numCycles;
        int                         printFrequency;
        int                         sampleFrequency;

        std::ofstream               convergenceLog;
        std::vector<TreeSamples*>   samples;
        std::set<int>               subsetIndices;

        std::vector<LikelihoodCalculator*> allocatedCalculators;
        std::vector<LikelihoodCalculator*> calculatorPool;
};

#endif
