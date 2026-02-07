#ifndef TreeList_hpp
#define TreeList_hpp

#include "BitSet.hpp"
#include <unordered_map>
#include <cstdint>
class Tree;

struct TreeInfo {

        TreeInfo(Tree* t, double x, bool tf) : lnL(x) {
        
            // Pack the boolean into the least significant bit of the tree pointer
            // (Tree objects are guaranteed to be aligned to at least 2-byte boundaries)
            uintptr_t ptr = reinterpret_cast<uintptr_t>(t);
            if (tf) 
                ptr |= 1;
            packedTreePtr = ptr;
        }
        
        Tree* getTree(void) const { 
        
            return reinterpret_cast<Tree*>(packedTreePtr & ~1ULL); 
        }
        
        bool isLikelihoodCalculated(void) const { 
        
            return packedTreePtr & 1; 
        }
        
        void setLikelihoodCalculated(bool tf) {
        
            if (tf) 
                packedTreePtr |= 1;
            else 
                packedTreePtr &= ~1ULL;
        }
        
        double lnL;
        
    private:
        uintptr_t packedTreePtr;  // tree pointer + boolean flag packed in LSB
};

typedef std::unordered_map<uint64_t, TreeInfo> TreeMap;
typedef TreeMap::iterator TreeMapIter;



class TreeList {

    public:
                    TreeList(size_t expectedSize = 1000000);
                   ~TreeList(void);
        double      lnLikelihood(Tree* t);
        void        addTree(Tree* t);
        void        addTree(Tree* t, double x);
        Tree*       getTree(uint64_t treeHash);
        TreeInfo&   getTreeInfo(uint64_t treeHash);
        void        print(void);
        void        reserve(size_t capacity);  // Method to reserve additional capacity
        size_t      size(void) { return map.size(); }
    
    private:
        TreeMap     map;
};

#endif 
