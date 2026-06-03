#ifndef TreeConvergenceDiagnostics_hpp
#define TreeConvergenceDiagnostics_hpp

#include <cstdint>
#include <ostream>
#include <unordered_map>
#include <vector>
#include "TreeCache.hpp"
#include "TreeSamples.hpp"

class TreeConvergenceDiagnostics {

    public:
                        TreeConvergenceDiagnostics(void) = delete;
        explicit        TreeConvergenceDiagnostics(TreeCache* tc);

        static void     writeStatsHeader(std::ostream& os);
        void            writeStatsLine(std::ostream& os, const std::vector<TreeSamples*>& sampleVec) const;

    private:
        struct ChainStats {
            size_t      numSamples = 0;
            size_t      numUniqueTrees = 0;
            double      discoveredMass = 0.0;
            double      credible95MassDiscovered = 0.0;
            double      credible95TreeCoverage = 0.0;
            double      top5Coverage = 0.0;
            double      top10Coverage = 0.0;
            double      top50Coverage = 0.0;
            double      tvExact = 0.0;
            double      jsExact = 0.0;
            double      meanTruePosteriorProbability = 0.0;
            double      currentTreePosteriorProbability = 0.0;
            double      currentTreeRank = 0.0;
            double      mapFirstHit = 0.0;
            double      tau50 = 0.0;
            double      tau90 = 0.0;
            double      tau95 = 0.0;
            bool        mapWasFound = false;
            bool        tau50WasReached = false;
            bool        tau90WasReached = false;
            bool        tau95WasReached = false;
        };

        struct SummaryStats {
            double      mean = 0.0;
            double      se = 0.0;
            double      n = 0.0;
        };

        ChainStats      computeChainStats(const TreeSamples* samples) const;
        double          empiricalProbability(const TreeSamples* samples, uint64_t hash) const;
        int             firstHitCycle(const std::vector<uint64_t>& trace, uint64_t hash) const;
        double          fractionFound(const TreeCountMap& counts, size_t firstRank, size_t lastRank) const;
        SummaryStats    summarize(const std::vector<double>& values) const;
        SummaryStats    summarizeConditional(const std::vector<double>& values, const std::vector<bool>& include) const;
        void            writeMeanSe(std::ostream& os, const SummaryStats& s) const;

        TreeCache*                                      treeCache;
        std::vector<TreeInfo*>                          posteriorRank;
        std::unordered_map<uint64_t,int>                rankByHash;
        std::unordered_map<uint64_t,bool>               inCredibleSet95;
        double                                          credible95Mass;
        size_t                                          numCredibleSet95Trees;
};

#endif
