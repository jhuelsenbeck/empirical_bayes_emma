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
        void            writeStatsLine(std::ostream& os, const std::vector<TreeSamples*>& sampleVec) const;

    private:
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
