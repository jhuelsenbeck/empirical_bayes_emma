#ifndef TreeNeighborhood_hpp
#define TreeNeighborhood_hpp

#include <cstdint>
#include <unordered_map>
#include <vector>
class RandomVariable;
class Tree;
class TreeList;

typedef std::vector<std::tuple<uint64_t,Tree*,double>> NeighborValues;
typedef std::vector<uint64_t> TreeHashVec;
typedef std::unordered_map<uint64_t,TreeHashVec> TreeNeighbors;


class TreeNeighborhood {

    public:
                                TreeNeighborhood(void) = delete;
                                TreeNeighborhood(TreeList* tl);
        virtual                ~TreeNeighborhood(void);
        void                    clear(void) { neighborhood.clear(); }
        void                    getNeighbors(uint64_t treeHash, NeighborValues& neighbors);
        void                    getNeighbors(uint64_t treeHash, TreeHashVec& vec);
        void                    getNeighbors(Tree* tree, NeighborValues& neighbors);
        virtual void            getNeighbors(uint64_t treeHash, NeighborValues& neighbors, uint64_t defaultTree);
        size_t                  size(void) { return neighborhood.size(); }

    protected:
        virtual void            generateNeighbors(Tree* tree, NeighborValues& neighbors) = 0;
        virtual void            generateNeighbors(uint64_t treeHash, NeighborValues& neighbors) = 0;
        TreeList*               treeList;
        TreeNeighbors           neighborhood;
};

class TreeNeighborhoodNni : public TreeNeighborhood {

    public:
                                TreeNeighborhoodNni(void) = delete;
                                TreeNeighborhoodNni(TreeList* tl);
    
    protected:
        void                    generateNeighbors(Tree* tree, NeighborValues& neighbors);
        void                    generateNeighbors(uint64_t treeHash, NeighborValues& neighbors);
};

class TreeNeighborhoodNni2 : public TreeNeighborhoodNni {

    public:
                                TreeNeighborhoodNni2(void) = delete;
                                TreeNeighborhoodNni2(TreeList* tl);
    
    private:
        void                    generateNeighbors(uint64_t treeHash, NeighborValues& neighbors);
};

class TreeNeighborhoodTbr : public TreeNeighborhood {

    public:
                                TreeNeighborhoodTbr(void) = delete;
                                TreeNeighborhoodTbr(TreeList* tl);
    
    protected:
        bool                    checkTree(Tree* t);
        void                    generateNeighbors(Tree* tree, NeighborValues& neighbors);
        void                    generateNeighbors(uint64_t treeHash, NeighborValues& neighbors);
};

class TreeNeighborhoodTbrRandom : public TreeNeighborhood {

    public:
                                TreeNeighborhoodTbrRandom(void) = delete;
                                TreeNeighborhoodTbrRandom(TreeList* tl, int nn, RandomVariable* r);
        using TreeNeighborhood::getNeighbors;
        void                    getNeighbors(uint64_t treeHash, NeighborValues& neighbors, uint64_t defaultTree) override;
    
    protected:
        bool                    checkTree(Tree* t);
        void                    generateNeighbors(Tree* tree, NeighborValues& neighbors) override;
        void                    generateNeighbors(uint64_t treeHash, NeighborValues& neighbors) override;
        RandomVariable*         rng;
        int                     numNeighbors;
};


#endif
