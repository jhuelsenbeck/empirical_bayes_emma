#ifndef TreeSamples_hpp
#define TreeSamples_hpp

#include <cstdint>
#include <iosfwd>
#include <unordered_map>
#include <vector>
#include "TreeCache.hpp"

typedef std::unordered_map<uint64_t, int> TreeCountMap;


class TreeSamples {

    public:
                                        TreeSamples(void) = delete;
                                        TreeSamples(TreeCache* tc);
        void                            clear(void);
        static void                     compareSamples(std::vector<TreeSamples*>& sampleVec);
        size_t                          getNumSamples(void) const { return (size_t)numSamples; }
        const TreeCountMap&             getTreeCounts(void) const { return treeCounts; }
        double                          getTreeProbability(uint64_t treeHash) const;
        const std::vector<uint64_t>&    getTrees(void) const { return trees; }
        void                            print(void) const;
        static void                     combinedPrint(std::vector<TreeSamples*>& sampleVec);
        void                            reserve(size_t n) { trees.reserve(n); }
        void                            sampleTree(uint64_t treeHash);
        static void                     writeStatsHeader(std::ostream& os, size_t numChains);
        static void                     writeStatsLine(std::ostream& os, std::vector<TreeSamples*>& sampleVec);

    private:
        static double                   computeIndicatorESS(const std::vector<uint64_t>& trace, uint64_t target);
        TreeCache*                      treeCache;
        int                             numSamples;
        TreeCountMap                    treeCounts;
        std::vector<uint64_t>           trees;
};

#endif
