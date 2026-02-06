#ifndef BitSetFactory_hpp
#define BitSetFactory_hpp

#include <vector>
class BitSet;



class BitSetFactory {

    public:
        static BitSetFactory&   getFactory(void) 
                                    {
                                    static BitSetFactory bsf;
                                    return bsf;
                                    }
        BitSet*                 getBitSet(void);
        int                     getNumAllocated(void) { return static_cast<int>(allocated.size()); }
        int                     getNumInPool(void) { return static_cast<int>(pool.size()); }
        void                    initialize(int nt);
        void                    returnToPool(BitSet* bs);
    
    private:
                                BitSetFactory(void);
                               ~BitSetFactory(void);
                                BitSetFactory(const BitSetFactory& tp) = delete;
        int                     numTaxa;
        bool                    isInitialized;
        std::vector<BitSet*>    pool;
        std::vector<BitSet*>    allocated;
};

#endif
