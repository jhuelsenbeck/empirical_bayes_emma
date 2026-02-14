#ifndef TreeSpace_hpp
#define TreeSpace_hpp

#include <set>
#include <unordered_map>
#include <vector>
class TreeList;
class TreeNeighborhood;

struct TreeSpaceNode {

    uint64_t                    treeHash;
    float                       lnL;
    int                         peakId;
    int                         distance;
    bool                        isPeak;
    std::set<TreeSpaceNode*>    neighbors;
};
struct PeakInfo {

    uint64_t                    treeHash;
    float                       lnL;
    int                         basinSize;
    int                         closestPeakDistance;
    uint64_t                    closestPeak;
    double                      averageDistanceToPeaks;
};
typedef std::unordered_map<uint64_t,TreeSpaceNode*> TreeNodesMap;



class TreeSpace {

    public:
                                TreeSpace(void) = delete;
                                TreeSpace(TreeList* tl);
                               ~TreeSpace(void);
        void                    characterize(void);
        TreeSpaceNode*          getTree(uint64_t treeHash);
    
    private:
        void                    adjacentTreeDistances(uint64_t treeHash);
        void                    calculateDistances(uint64_t treeHash);
        void                    characterizeBasins(std::unordered_map<uint64_t,int>& basins);
        TreeSpaceNode*          findBestNeighbor(TreeSpaceNode* n);
        uint64_t                steepestAscent(uint64_t startTree);
        TreeList*               treeList;
        TreeSpaceNode*          root;
        TreeNodesMap            treeNodes;
};

#endif
