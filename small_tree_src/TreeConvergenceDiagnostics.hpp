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

        static void     writeStatsHeader(std::ostream& os, size_t numReplicates = 0);
        void            writeStatsLine(std::ostream& os, const std::vector<TreeSamples*>& sampleVec);

    private:
                        // Per-chain running totals, advanced over the samples drawn since the previous
                        // call. Earlier versions recomputed every statistic from the start of each trace
                        // at each of the ~60 points a run is logged, and re-sorted the whole tree space
                        // each time as well; the object now lives for the whole run.
        struct ChainCursor {
            size_t                          processed = 0;
            std::vector<unsigned char>      visited;          // by TreeInfo::index: seen yet?
            double                          discoveredMass = 0.0;
            double                          credible95MassDiscovered = 0.0;
            bool                            tau95WasReached = false;
            double                          tau95 = 0.0;
            double                          sumP = 0.0;
            int                             mapFirstHitCycle = -1;
            uint64_t                        lastHash = 0;
            const TreeInfo*                 lastInfo = nullptr;
        };
        std::vector<ChainCursor>            cursors;
        void            advance(ChainCursor& cur, const TreeSamples* samples);
        struct ChainStats {
            size_t      numSamples = 0;

            double      discoveredMass = 0.0;
            double      undiscoveredMass = 1.0;
            double      credible95MassDiscovered = 0.0;

            double      tau95 = 0.0;
            bool        tau95WasReached = false;

            double      meanTruePosteriorProbability = 0.0;

            // -1 means the MAP tree has not been hit by the current cycle.
            int         mapFirstHitCycle = -1;
        };

        struct SummaryStats {
            double      mean = 0.0;
            double      se = 0.0;
            double      n = 0.0;
        };

        ChainStats      computeChainStats(const TreeSamples* samples) const;
        int             firstHitCycle(const std::vector<uint64_t>& trace, uint64_t hash) const;

        SummaryStats    summarize(const std::vector<double>& values) const;
        SummaryStats    summarizeConditional(const std::vector<double>& values,
                                             const std::vector<bool>& include) const;

        void            writeMeanSe(std::ostream& os, const SummaryStats& s) const;

        TreeCache*                                      treeCache;
        std::vector<TreeInfo*>                          posteriorRank;
        std::unordered_map<uint64_t,bool>               inCredibleSet95;

        double                                          credible95Mass;
        size_t                                          numCredibleSet95Trees;
};

#endif
