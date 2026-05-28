#ifndef TreeNeighborGenerator_hpp
#define TreeNeighborGenerator_hpp

#include <vector>
#include "TreeCache.hpp"
class RandomVariable;
class Tree;
struct TreeInfo;



class TreeNeighborGenerator {

    public:
                        TreeNeighborGenerator(void) = delete;
                        TreeNeighborGenerator(TreeCache* tc);
        bool            checkTree(Tree* t);
        virtual void    generateNeighbors(Tree* tree, std::vector<TreeInfo*>& neighbors) = 0;
        virtual void    generateNeighbors(Tree* tree, std::vector<TreeInfo*>& neighbors, RandomVariable* rng) = 0;
    
    protected:
        TreeCache*      treeCache;
};

class TreeNeighborGeneratorNNI : public TreeNeighborGenerator {

    public:
                        TreeNeighborGeneratorNNI(void) = delete;
                        TreeNeighborGeneratorNNI(TreeCache* tc);
        void            generateNeighbors(Tree* tree, std::vector<TreeInfo*>& neighbors);
        void            generateNeighbors(Tree* tree, std::vector<TreeInfo*>& neighbors, RandomVariable*);
};

class TreeNeighborGeneratorTBR : public TreeNeighborGenerator {

    public:
                        TreeNeighborGeneratorTBR(void) = delete;
                        TreeNeighborGeneratorTBR(TreeCache* tc);
        void            generateNeighbors(Tree* tree, std::vector<TreeInfo*>& neighbors);
        void            generateNeighbors(Tree* tree, std::vector<TreeInfo*>& neighbors, RandomVariable*);
};

class TreeNeighborGeneratorRandomTBR : public TreeNeighborGenerator {

    public:
                        TreeNeighborGeneratorRandomTBR(void) = delete;
                        TreeNeighborGeneratorRandomTBR(TreeCache* tc);
        void            generateNeighbors(Tree* tree, std::vector<TreeInfo*>& neighbors);
        void            generateNeighbors(Tree* tree, std::vector<TreeInfo*>& neighbors, RandomVariable* rng);
};

#endif
