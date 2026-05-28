#ifndef TreeSpace_hpp
#define TreeSpace_hpp

#include <set>
#include <unordered_map>
#include <vector>
#include "TreeCache.hpp"

struct TreeSpaceNode {

    uint64_t                    treeHash;
    float                       lnL;
    int                         peakId;
    TreeSpaceNode*              peakPtr;
    int                         distance;
    bool                        isPeak;
    std::set<TreeSpaceNode*>    neighbors;
};
struct PeakInfo {

    uint64_t                    treeHash;
    float                       lnL;
    float                       minLnL;
    int                         peakId;
    int                         basinSize;
};
class Peak;
class TreeSamples;
typedef std::unordered_map<uint64_t,TreeSpaceNode*> TreeNodesMap;
typedef std::unordered_map<uint64_t,double> TreeProbMap;
typedef std::unordered_map<uint64_t,Peak*> PeakMap;



class TreeSpace {

    public:
                                TreeSpace(void) = delete;
                                TreeSpace(TreeCache* tc);
                               ~TreeSpace(void);
        void                    characterize(void);
        Peak*                   findPeak(uint64_t treeHash);
        int                     findPeakIdForTreeWithHash(uint64_t treeHash);
        Peak*                   findPeakWithId(int id);
        TreeSpaceNode*          getTree(uint64_t treeHash);
        double                  getTreeProbabiity(uint64_t treeHash);
        int                     graphDistance(const TreeSpaceNode* a, const TreeSpaceNode* b);
        void                    printPosterior(void);
        void                    printPosterior(std::string fileName);
        void                    printPosterior(TreeSamples* samples);
    
    private:
        void                    adjacentTreeDistances(uint64_t treeHash);
        void                    calculateDistances(uint64_t treeHash);
        void                    characterizeBasins(std::unordered_map<uint64_t,int>& basins);
        TreeSpaceNode*          findBestNeighbor(TreeSpaceNode* n);
        uint64_t                steepestAscent(uint64_t startTree);
        TreeCache*              treeCache;
        TreeNodesMap            treeNodes;
        PeakMap                 peaks;
        TreeProbMap             treeProbabilities;
};

#endif
