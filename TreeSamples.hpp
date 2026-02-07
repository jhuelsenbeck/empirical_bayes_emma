#ifndef TreeSamples_hpp
#define TreeSamples_hpp

#include <unordered_map>
class TreeList;

typedef std::unordered_map<uint64_t, int> TreeCountMap;


class TreeSamples {

    public:
                        TreeSamples(void) = delete;
                        TreeSamples(TreeList* tl);
        void            print(void);
        void            sampleTree(uint64_t treeHash);
    
    private:
        TreeList*       treeList;
        int             numSamples;
        TreeCountMap    treeCounts;
};

#endif
