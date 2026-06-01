#ifndef TreeConvergenceDiagnostics_hpp
#define TreeConvergenceDiagnostics_hpp

#include <iosfwd>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include "TreeCache.hpp"
#include "TreeSamples.hpp"

class TreeConvergenceDiagnostics {

    public:
        TreeConvergenceDiagnostics(void) = delete;
        explicit TreeConvergenceDiagnostics(TreeCache* tc);

        static void writeStatsHeader(std::ostream& os, size_t numChains);
        void        writeStatsLine(std::ostream& os, const std::vector<TreeSamples*>& sampleVec) const;

    private:
        struct ChainStats {
            size_t numSamples = 0;
            size_t numUniqueTrees = 0;
            double visitedPosteriorMass = 0.0;
            double credibleSet95Coverage = 0.0;
            double totalVariation = 0.0;
            double klEmpiricalToTrue = 0.0;
            double klTrueToEmpirical = 0.0;
            double jsDivergence = 0.0;
            double meanTruePosteriorProbability = 0.0;
            double meanLnTruePosteriorProbability = 0.0;
            uint64_t currentTreeHash = 0;
            double currentTreePosteriorProbability = 0.0;
            int currentTreeRank = 0;
        };

        struct PairStats {
            double meanPairwiseTv = 0.0;
            double maxPairwiseTv = 0.0;
            double meanPairwiseJs = 0.0;
            double maxPairwiseJs = 0.0;
        };

        ChainStats computeChainStats(const TreeSamples* samples) const;
        PairStats  computePairStats(const std::vector<TreeSamples*>& sampleVec) const;
        double     empiricalProbability(const TreeSamples* samples, uint64_t hash) const;
        double     safeLog(double x) const;

        TreeCache* treeCache;
        std::vector<TreeInfo*> posteriorRank;
        std::unordered_map<uint64_t,int> rankByHash;
        std::unordered_map<uint64_t,bool> inCredibleSet95;
        size_t numCredibleSet95Trees;
};

#endif
