#include "BitSet.hpp"
#include "BitSetFactory.hpp"
#include "Msg.hpp"



BitSetFactory::BitSetFactory(void) : isInitialized(false) {

}

BitSetFactory::~BitSetFactory(void) {

    for (size_t i=0; i<allocated.size(); i++)
        delete allocated[i];
}

BitSet* BitSetFactory::getBitSet(void) {

    if (isInitialized == false)
        Msg::error("BitSet Factory is not initialized");
        
    if (pool.empty() == true)
        {
        // if the pool is empty, we allocate a new block of nodes
        BitSet* newBitSet = new BitSet(numTaxa);
        allocated.push_back(newBitSet);
        pool.push_back(newBitSet);
        }
    
    // return a node from the pool, remembering to remove it from the pool
    BitSet* bs = pool.back();
    pool.pop_back();
    return bs;
}

void BitSetFactory::initialize(int nt) {

    if (isInitialized == true)
        Msg::error("BitSet Factory is already initialized");
    numTaxa = nt;
    isInitialized = true;
}

void BitSetFactory::returnToPool(BitSet* bs) {

    bs->unset();
    pool.push_back(bs);
}
