#ifndef TreeNeighbors_hpp
#define TreeNeighbors_hpp

#include "TreeCache.hpp"
class Tree;



class TreeNeighbors {

    public:
                                TreeNeighbors(void) = delete;
                                TreeNeighbors(TreeCache* tc, TreeNeighborGenerator* ng, int nt);
        std::vector<TreeInfo*>& neighbors(Tree* t);
        std::vector<TreeInfo*>& neighbors(uint64_t treeHash);
    
    private:
        TreeCache*              treeCache;
        TreeNeighborGenerator*  neighborGenerator;
        int                     numTaxa;
};

#endif
