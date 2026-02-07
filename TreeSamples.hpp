#ifndef TreeSamples_hpp
#define TreeSamples_hpp

#include <unordered_map>

typedef std::unordered_map<uint64_t, int> TreeCountMap;


class TreeSamples {

    public:
                        TreeSamples(void);
        void            print(void);
        void            sampleTree(uint64_t treeHash);
    
    private:
        int             numSamples;
        TreeCountMap    treeCounts;
};

#endif
