#ifndef Mcmc_hpp
#define Mcmc_hpp

#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "RandomVariable.hpp"
#include "Threads.hpp"
#include "TreeCache.hpp"

class Alignment;
class LikelihoodCalculator;
class TreeSamples;
class TreeConvergenceDiagnostics;

class Mcmc {

    public:
                                    Mcmc(RandomVariable* r, TreeCache* tc, Alignment* a, bool tf=false, std::string cfn="", ThreadPool* tp=nullptr);
                                   ~Mcmc(void);
        void                        run(std::string label, int numRuns);
        void                        run(std::string label, double power, int nNeighbors);
        void                        run(std::string label, double power, int nNeighbors, int numRuns);
        void                        run(std::string label, double power, int nNeighbors, int numRuns, int numHeatedChains);

    private:
        std::pair<int,int>          chooseChains(RandomVariable* r, int numChains);
        uint64_t                    chooseGibbs(RandomVariable* r, std::vector<std::pair<uint64_t,double>>& probs, const std::vector<double>& cdf);
        TreeInfo*                   chooseInitialTreeInfo(void);
        TreeInfo*                   chooseTreeInfo(RandomVariable* r, TreeInfo* currentInfo, double& proposalProbability);
        TreeInfo*                   chooseTreeInfo(RandomVariable* r, std::vector<int>& subset, TreeInfo* currentInfo, double& proposalProbability, int n, double power);
        int                         coldChainIndex(std::vector<int>& chainIndices);
        void                        deleteSamplesAndPartitions(void);
        double                      findTreeProbability(TreeInfo* fromInfo, const TreeInfo* toInfo);
        double                      findTreeProbability(RandomVariable* r, std::vector<int>& subset, TreeInfo* fromInfo, const TreeInfo* toInfo, int n, double power);
        void                        forEachRun(int numRuns, const std::function<void(int)>& body);
        std::vector<RandomVariable> makeRunGenerators(int numRuns);
        uint32_t                    segmentEnd(uint32_t first);
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
        ThreadPool*                 threadPool;                      // replicate runs in parallel when set
        Alignment*                  alignment;
        TreeCache*                  treeCache;
        bool                        expandedOutput;
        std::string                 convergenceLogFileName;
        double                      temperature;
        int                         numCycles;
        int                         printFrequency;
        int                         sampleFrequency;

        std::ofstream               convergenceLog;
        std::unique_ptr<TreeConvergenceDiagnostics> diagnostics;     // built once per run, with the log
        std::vector<TreeSamples*>   samples;
        std::vector<int>            subsetIndices;

        std::vector<LikelihoodCalculator*> allocatedCalculators;
        std::vector<LikelihoodCalculator*> calculatorPool;
};

#endif
