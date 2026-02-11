#ifndef TreeList_hpp
#define TreeList_hpp

#include "BitSet.hpp"
#include "TreeInfo.hpp"
#include <cstdint>
#include <unordered_map>
class Tree;

typedef std::unordered_map<uint64_t, TreeInfo> TreeMap;
typedef TreeMap::iterator TreeMapIter;



class TreeList {

    public:
                                    TreeList(std::vector<std::string> tNames, size_t expectedSize = 1000000);
                                   ~TreeList(void);
        std::vector<std::string>&   getTaxonNames(void) { return taxonNames; }
        void                        addTree(Tree* t);
        void                        addTree(Tree* t, double x);
        double                      distance(uint64_t t1, uint64_t t2);
        Tree*                       getTree(uint64_t treeHash);
        TreeInfo&                   getTreeInfo(uint64_t treeHash);
        TreeMap&                    getTreeList(void) { return map; }
        bool                        isTreeInList(uint64_t treeHash);
        double                      lnLikelihood(Tree* t);
        void                        print(void);
        void                        reserve(size_t capacity);  // Method to reserve additional capacity
        size_t                      size(void) { return map.size(); }
    
    private:
        TreeMap                     map;
        std::vector<std::string>    taxonNames;
};

#endif 
