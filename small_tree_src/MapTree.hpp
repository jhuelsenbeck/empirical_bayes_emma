#ifndef MapTree_hpp
#define MapTree_hpp

#include <set>
#include <string>
#include <unordered_map>
class Tree;
class TreeCache;
struct TreeInfo;
typedef std::unordered_map<uint16_t,std::set<TreeInfo*>> PartitionMap;



class MapTree {

    public:
                            MapTree(void) = delete;
                            MapTree(TreeCache* c);
        uint64_t            getMapTree(void) { return mapHash; }
        const PartitionMap& getPartitions(void) const { return mapPartitions; }
        double              partitionProbability(uint16_t part);
        std::string         partitionString(uint16_t part);
        void                print(void);
    
    private:
        void                findMapTree(void);
        void                findPartitions(void);
        void                printPartition(uint16_t part);
        TreeCache*          treeCache;
        uint64_t            mapHash;
        Tree*               mapTree;
        PartitionMap        mapPartitions;
        int                 numTaxa;
};

#endif
