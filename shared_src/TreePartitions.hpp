#ifndef TreePartitions_hpp
#define TreePartitions_hpp

#include <iosfwd>
#include <map>
class BitSet;
class Tree;

typedef std::map<BitSet*,uint32_t, CompBitSet> PartitionMap;



class TreePartitions {

    public:
                        TreePartitions(void) = delete;
                        TreePartitions(int n);
                       ~TreePartitions(void);
        void            addTree(Tree* t);
        static void     comparePartitions(std::vector<TreePartitions*>& parts, bool loudOutput);
        void            print(void);
        size_t          size(void) { return taxonBipartitions.size(); }
        static void     writeStatsHeader(std::ostream& os);
        static void     writeStatsLine(std::ostream& os, std::vector<TreePartitions*>& parts);    
        
    private:
        int             numTaxa;
        uint32_t        count;
        PartitionMap    taxonBipartitions;
};

#endif
