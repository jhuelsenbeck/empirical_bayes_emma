#ifndef TreeSamples_hpp
#define TreeSamples_hpp

#include <map>
#include <unordered_map>
class TreeList;

typedef std::unordered_map<uint64_t, int> TreeCountMap;


class TreeSamples {

    public:
                        TreeSamples(void) = delete;
                        TreeSamples(TreeList* tl);
        void            print(std::map<uint64_t,std::pair<double,double>>& treeProbabilities);
        void            sampleTree(uint64_t treeHash);
    
    private:
        TreeList*       treeList;
        int             numSamples;
        TreeCountMap    treeCounts;
};

#endif
