#ifndef CompactTree_hpp
#define CompactTree_hpp

#include <cstdint>
#include <string>
#include <vector>
class Node;
class Tree;



class BitWriter {

    public:
        void write(uint32_t value, uint8_t nbits) {
        
            for (uint8_t i = 0; i < nbits; ++i) 
                {
                if (pos_ / 8 >= buf_.size())
                    buf_.push_back(0);
                if ((value >> i) & 1)
                    buf_[pos_ / 8] |= (1u << (pos_ % 8));
                ++pos_;
                }
        }
        
        const std::vector<uint8_t>& data(void) const { return buf_; }
        uint32_t bitPos(void) const { return pos_; }
    
    private:
        std::vector<uint8_t> buf_;
        uint32_t             pos_ = 0;
};

class BitReader {

    public:
                    BitReader(const uint8_t* data) : data_(data) {}
        uint32_t    read(uint8_t nbits) {
        
                        uint32_t val = 0;
                        for (uint8_t i = 0; i < nbits; ++i)
                            {
                            val |= (uint32_t((data_[pos_ / 8] >> (pos_ % 8)) & 1) << i);
                            ++pos_;
                            }
                        return val;
                    }
        uint32_t    peek(uint8_t nbits) const {
        
                        uint32_t val = 0;
                        for (uint8_t i = 0; i < nbits; ++i)
                            val |= (uint32_t((data_[(pos_ + i) / 8] >> ((pos_ + i) % 8)) & 1) << i);
                        return val;
                    }
    
    private:
        const uint8_t*  data_;
        uint32_t        pos_ = 0;
};



// Compact binary encoding of a phylogenetic tree rooted on tip 0.
// Stores topology only (no branch lengths).
//
// Topology encoding (bit-packed):
//   Header:  10 bits — number of bits per taxon index
//   Codes:   00 = open clade (internal node)
//            01 = close clade
//            10 = end of tree (semicolon)
//            11 = taxon leaf (followed by taxon index in 'taxon_bits' bits)
//
// Tip 0 is always the root and is not encoded.
//
// Memory per tree (topology only):
//   N=30:   ~43 bytes    (vs ~8 KB for full Tree)
//   N=100:  ~163 bytes   (vs ~28 KB)
//   N=1000: ~2 KB        (vs ~280 KB)

class CompactTree {

    public:
                                CompactTree(void) = default;
        
                                // encode a Tree into compact representation. Does NOT take ownership of t.
        static CompactTree      encode(Tree* t);
        
                                // decode back into a Newick string (caller uses Tree's Newick constructor)
        std::string             toNewick(std::vector<std::string>& taxonNames) const;
        
                                // size in bytes of this compact representation
        size_t                  sizeInBytes(void) const;
    
    private:
        static uint8_t          bitsNeeded(int n);
        static void             encodeNode(Node* p, BitWriter& bw, uint8_t taxonBits);
        std::string             decodeNode(BitReader& br, uint8_t taxonBits, std::vector<std::string>& taxonNames) const;
    
        std::vector<uint8_t>    topo_;          // bit-packed topology
        uint32_t                topoBits_ = 0;  // actual bits used
        uint8_t                 taxonBits_ = 0; // bits per taxon index
        uint16_t                numTips_ = 0;
};

#endif
