#include "CompactTree.hpp"
#include "Node.hpp"
#include "Tree.hpp"



CompactTree::CompactTree(const CompactTree& ct) : BitSet(ct) {

}

int CompactTree::bitsPerIndex(int n) {

    int bits = 0;
    do
        {
        ++bits;
        n >>= 1;
        } while (n);
    return bits;
}

void CompactTree::writeCode(size_t& pos, Code c) {

    uint8_t bits = static_cast<uint8_t>(c);
    if (bits & 1)
        set(pos);
    if (bits & 2)
        set(pos + 1);
    pos += 2;
}

CompactTree::Code CompactTree::readCode(size_t& pos) const {

    uint8_t bits = (isSet(pos)     ? 1 : 0)
                 | (isSet(pos + 1) ? 2 : 0);
    pos += 2;
    return static_cast<Code>(bits);
}

void CompactTree::writeIndex(uint32_t idx, size_t& pos, int taxonBits) {

    // emit MSB first; the BitSet was sized and zero-initialized, so we only
    // need to set the 1 bits
    for (int i = 0; i < taxonBits; ++i, ++pos)
        {
        if (idx & (uint32_t(1) << (taxonBits - 1 - i)))
            set(pos);
        }
}

uint32_t CompactTree::readIndex(size_t& pos, int taxonBits) const {

    uint32_t val = 0;
    for (int i = 0; i < taxonBits; ++i, ++pos)
        {
        if (isSet(pos))
            val |= (uint32_t(1) << (taxonBits - 1 - i));
        }
    return val;
}

void CompactTree::writeSubtree(Node* p, Node* root, size_t& pos, int taxonBits) {

    if (p->getIsTip())
        {
        writeCode(pos, Code::Tip);
        writeIndex(p->getIndex(), pos, taxonBits);
        return;
        }

    // internal: open, then each child followed by a Close (sibling separator
    // for non-last children, parent-pop for the last)
    writeCode(pos, Code::Open);
    bool first = true;
    for (Node* d = p->getFirstDescendant(); d != nullptr; d = d->getNextSibling())
        {
        if (!first)
            writeCode(pos, Code::Close);            // sibling separator
        first = false;
        writeSubtree(d, root, pos, taxonBits);
        }
    writeCode(pos, Code::Close);                    // close

    // root case: tip 0 is appended as the topmost's last sibling in the flat form
    if (p == root->getFirstDescendant())
        {
        writeCode(pos, Code::Tip);
        writeIndex(root->getIndex(), pos, taxonBits);
        }
}

CompactTree CompactTree::encode(Tree* t) {

    int    taxonBits = bitsPerIndex(t->getNumTips());
    int    n         = t->getNumTips();

    // Bit budget for a binary tree on n tips: 2 bits per "(", ")", and ","
    // structural code, plus (2 + taxonBits) bits per tip emission, plus 2
    // bits for the trailing End terminator. This formula slightly over-
    // allocates (by 2 bits) for the rooted-on-tip-0 layout — harmless.
    int    numLeftParens   = n - 2;
    int    numRightParens  = n - 2;
    int    numCommas       = n - 1;
    size_t numBits = 2 * (numLeftParens + numRightParens + numCommas)
                   + static_cast<size_t>(n) * (2 + taxonBits)
                   + 2;

    CompactTree ct(numBits);
    size_t pos = 0;
    ct.writeSubtree(t->getRoot()->getFirstDescendant(), t->getRoot(), pos, taxonBits);
    ct.writeCode(pos, Code::End);

    return ct;
}

size_t CompactTree::sizeInBytes(void) const {

    return getNumUnints() * sizeof(unsigned) + sizeof(CompactTree);
}
