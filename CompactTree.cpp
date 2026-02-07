#include "CompactTree.hpp"
#include "Node.hpp"
#include "Tree.hpp"



uint8_t CompactTree::bitsNeeded(int n) {

    if (n <= 2) return 1;
    uint8_t bits = 0;
    int v = n - 1;
    while (v > 0) 
        {
        bits++;
        v >>= 1;
        }
    return bits;
}

void CompactTree::encodeNode(Node* p, BitWriter& bw, uint8_t taxonBits) {

    if (p->getIsTip())
        {
        bw.write(0b11, 2);                             // taxon marker
        bw.write(static_cast<uint32_t>(p->getIndex()), taxonBits);
        }
    else
        {
        bw.write(0b00, 2);                             // open clade
        for (Node* d = p->getFirstDescendant(); d != nullptr; d = d->getNextSibling())
            encodeNode(d, bw, taxonBits);
        bw.write(0b01, 2);                             // close clade
        }
}

CompactTree CompactTree::encode(Tree* t) {

    CompactTree ct;
    ct.numTips_ = static_cast<uint16_t>(t->getNumTips());
    ct.taxonBits_ = bitsNeeded(ct.numTips_);
    
    BitWriter bw;
    
    // header: 10 bits for taxon bit width
    bw.write(ct.taxonBits_, 10);
    
    // encode from root's first descendant (root is tip 0, always implicit)
    Node* rootDesc = t->getRoot()->getFirstDescendant();
    encodeNode(rootDesc, bw, ct.taxonBits_);
    
    // semicolon
    bw.write(0b10, 2);
    
    ct.topo_ = bw.data();
    ct.topoBits_ = bw.bitPos();
    
    return ct;
}

std::string CompactTree::decodeNode(BitReader& br, uint8_t taxonBits, std::vector<std::string>& taxonNames) const {

    uint32_t code = br.read(2);
    
    if (code == 0b11)
        {
        // tip node
        uint32_t idx = br.read(taxonBits);
        return taxonNames[idx];
        }
    
    if (code == 0b00)
        {
        // internal node
        std::string result = "(";
        bool first = true;
        
        // decode children until close-paren
        while (br.peek(2) != 0b01)
            {
            if (!first) 
                result += ",";
            first = false;
            result += decodeNode(br, taxonBits, taxonNames);
            }
        br.read(2);  // consume the close-paren
        
        return result + ")";
        }
    
    // shouldn't reach here (codes 01, 10 handled by caller)
    return "";
}

std::string CompactTree::toNewick(std::vector<std::string>& taxonNames) const {

    BitReader br(topo_.data());
    
    // skip the 10-bit header (we already have taxonBits_)
    br.read(10);
    
    std::string newick = decodeNode(br, taxonBits_, taxonNames);
    // br should now be at the semicolon code (0b10)
    
    return newick + ";";
}

size_t CompactTree::sizeInBytes(void) const {

    return topo_.capacity() + sizeof(CompactTree);
}
