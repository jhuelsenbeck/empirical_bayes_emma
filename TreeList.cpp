#include <iomanip>
#include <iostream>
#include <set>
#include "BitSet.hpp"
#include "BitSetFactory.hpp"
#include "Msg.hpp"
#include "Node.hpp"
#include "Tree.hpp"
#include "TreeList.hpp"



TreeList::TreeList(std::vector<std::string> tNames, size_t expectedSize) : taxonNames(tNames) {

    map.reserve(expectedSize);
}

TreeList::~TreeList(void) {

    for (TreeMapIter it=map.begin(); it != map.end(); it++)
        delete it->second.getTree();
}

void TreeList::addTree(Tree* t) {

    map.emplace(t->getHash(), TreeInfo(t, 0.0, false));
}

void TreeList::addTree(Tree* t, double x) {

    map.emplace(t->getHash(), TreeInfo(t, x, false));
}

double TreeList::distance(uint64_t t1, uint64_t t2) {

    if (t1 == t2)
        return 0.0;
        
    BitSetFactory& bf = BitSetFactory::getFactory();
    
    // get information from both trees
    Tree* tree1 = getTree(t1);
    Tree* tree2 = getTree(t2);
    if (tree1 == nullptr || tree2 == nullptr)
        Msg::error("Could not find trees to compare when calculating tree-to-tree distance");
    int numNodes = tree1->getNumNodes();
    if (tree2->getNumNodes() != numNodes)
        Msg::error("Attempting to compare trees of different sizes");
    int numTips = tree1->getNumTips();
    if (tree2->getNumTips() != numTips)
        Msg::error("Attempting to compare trees of different sizes");

    // get non-trivial parititions for tree1
    std::vector<BitSet*> partitions1(numNodes);
    std::set<BitSet*,CompBitSet> nonTrivialPartitions1;
    for (size_t i=0; i<numNodes; i++)
        partitions1[i] = bf.getBitSet();
    for (size_t i=0; i<numTips; i++)
        partitions1[i]->set(i);
    std::vector<Node*>& dpSeq1 = tree1->getDownPassSequence();
    for (Node* p : dpSeq1)
        {
        if (p->getIsTip() == true)
            continue;
        if (p->getAncestor() == tree1->getRoot())
            continue;
            
        for (Node* d=p->getFirstDescendant(); d != nullptr; d=d->getNextSibling())
            *(partitions1[p->getIndex()]) |= *(partitions1[d->getIndex()]);
            
        nonTrivialPartitions1.insert(partitions1[p->getIndex()]);
        }

    // get non-trivial parititions for tree2
    std::vector<BitSet*> partitions2(numNodes);
    for (size_t i=0; i<numNodes; i++)
        partitions2[i] = bf.getBitSet();
    for (size_t i=0; i<numTips; i++)
        partitions2[i]->set(i);
    std::vector<Node*>& dpSeq2 = tree2->getDownPassSequence();
    int numDifferent = 0;
    for (Node* p : dpSeq2)
        {
        if (p->getIsTip() == true)
            continue;
        if (p->getAncestor() == tree1->getRoot())
            continue;
            
        for (Node* d=p->getFirstDescendant(); d != nullptr; d=d->getNextSibling())
            *(partitions2[p->getIndex()]) |= *(partitions2[d->getIndex()]);
            
        std::set<BitSet*,CompBitSet>::iterator it = nonTrivialPartitions1.find(partitions2[p->getIndex()]);
        if (it == nonTrivialPartitions1.end())
            numDifferent++;
        }
                
    for (BitSet* bs : partitions1)
        bf.returnToPool(bs);
    for (BitSet* bs : partitions2)
        bf.returnToPool(bs);

    return (double)(2 * numDifferent);
}

Tree* TreeList::getTree(uint64_t treeHash) {

    TreeMapIter it = map.find(treeHash);
    if (it == map.end())
        return nullptr;
    return it->second.getTree();
}

TreeInfo& TreeList::getTreeInfo(uint64_t treeHash) {

    TreeMapIter it = map.find(treeHash);
    if (it == map.end())
        Msg::error("Could not find tree in tree list");
    return it->second;
}

bool TreeList::isTreeInList(uint64_t treeHash) {

    TreeMapIter it = map.find(treeHash);
    if (it == map.end())
        return false;
    return true;
}

double TreeList::lnLikelihood(Tree* t) {

    TreeMap::iterator it = map.find(t->getHash());
    if (it == map.end())
        {
        map.emplace(t->getHash(), TreeInfo(t, 0.0, false));
        }
    
    return it->second.lnL;
}


void TreeList::print(void) {

    int i = 1;
    for (TreeMapIter it = map.begin(); it != map.end(); it++)
        std::cout << std::setw(6) << i++ << " " << std::setw(20) << it->first << " " << it->second.lnL << std::endl;
}

void TreeList::reserve(size_t capacity) {

    map.reserve(capacity);
}
