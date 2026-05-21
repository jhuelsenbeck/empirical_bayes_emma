#ifndef CompactTree_hpp
#define CompactTree_hpp

#include <cstdint>
#include <string>
#include <vector>
#include "BitSet.hpp"

class Node;
class Tree;



// Compact binary encoding of a phylogenetic tree rooted on tip 0.
// Stores topology only (no branch lengths).
//
// Inherits from BitSet for bit-packed storage. CompactTree adds no member
// state of its own — everything is delegated to BitSet — and owns the
// format via encode() / readCode() / readIndex() / bitsPerIndex(). Callers
// (e.g. Tree's constructor) walk the bit stream directly through these
// helpers; there is no string intermediate.
//
// Encoding (two-bit codes):
//   "00"  taxon leaf, followed by ceil(log2(numTips)) bits of taxon index
//         emitted MSB first
//   "01"  open internal node
//   "10"  close internal node — also serves as sibling separator: every
//         internal emits one trailing "10" after each child, so a node with
//         K children produces K instances of "10" before control returns to
//         its parent
//   "11"  end of stream
//
// The serialization is "flat": tip 0 is emitted as the last sibling of the
// topmost internal node. Receivers that want the original "rooted on tip 0"
// layout should call rerootOnTipZero() on the reconstructed Tree.

class CompactTree : public BitSet {

    public:
        enum class Code : uint8_t
            {
            Tip   = 0b00,
            Open  = 0b01,
            Close = 0b10,
            End   = 0b11
            };

                                CompactTree(void) = default;
        explicit                CompactTree(size_t n) : BitSet(n) {}

                                // Encode a Tree (assumed rooted on tip 0). Does NOT take ownership of t.
        static CompactTree      encode(Tree* t);
        static CompactTree*     encodeNew(Tree* t) { return new CompactTree(encode(t)); }

                                // Number of bits needed to represent a single taxon index for a
                                // tree with n tips.
        static int              bitsPerIndex(int n);

                                // Read the next two-bit code, advancing pos by 2.
        Code                    readCode(size_t& pos) const;

                                // Read a taxon index (taxonBits bits, MSB first) starting at pos,
                                // advancing pos by taxonBits.
        uint32_t                readIndex(size_t& pos, int taxonBits) const;

                                // Memory footprint of this representation.
        size_t                  sizeInBytes(void) const;

    private:
        void                    writeCode(size_t& pos, Code c);
        void                    writeIndex(uint32_t idx, size_t& pos, int taxonBits);
        void                    writeSubtree(Node* p, Node* root, size_t& pos, int taxonBits);
};

#endif
