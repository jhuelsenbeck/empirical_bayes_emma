#include <unordered_set>
#include "Msg.hpp"
#include "Node.hpp"
#include "Tree.hpp"
#include "TreeList.hpp"
#include "TreeNeighborhood.hpp"



TreeNeighborhood::TreeNeighborhood(TreeList* tl) : treeList(tl) {

}

TreeNeighborhood::~TreeNeighborhood(void) {

}

TreeNeighborhoodNni::TreeNeighborhoodNni(TreeList* tl) : TreeNeighborhood(tl) {

}

std::vector<uint64_t>& TreeNeighborhoodNni::getNeighbors(Tree* tree) {

    return getNeighbors(tree->getHash());
}

std::vector<uint64_t>& TreeNeighborhoodNni::getNeighbors(uint64_t treeHash) {

    TreeNeighbors::iterator it = nniNeighborhood.find(treeHash);
    if (it == nniNeighborhood.end())
        {
        std::vector<uint64_t> neighbors;
        generateNniNeighbors(treeHash, neighbors);
        nniNeighborhood.insert( std::make_pair(treeHash,neighbors) );
        it = nniNeighborhood.find(treeHash);
        return it->second;
        }
    return it->second;
}

void TreeNeighborhoodNni::generateNniNeighbors(uint64_t treeHash, std::vector<uint64_t>& neighbors) {

    Tree* tree = treeList->getTree(treeHash);
    if (tree == nullptr)
        Msg::error("Could not find tree in tree list");
        
    neighbors.resize(2*(tree->getNumTips()-3));
    const std::vector<Node*>& postOrder = tree->getDownPassSequence();
    Node* treeRoot = tree->getRoot();
    int k = 0;
    for (Node* p : postOrder)
        {
        if (p->getIsTip() == true)
            continue;
        if (p->getAncestor() == treeRoot)
            continue;

        // find area of rearrangment for NNI swap
        Node* pA = p->getAncestor();
        Node* d0 = p->getFirstDescendant();
        Node* d1 = d0->getNextSibling();
        Node* pS = p->getSister();
        if (d0 == nullptr || d1 == nullptr || pS == nullptr)
            Msg::error("Could not find nodes for NNI rearrangement");
        size_t p_offset  = p->getOffset();
        size_t pA_offset = pA->getOffset();
        size_t d0_offset = d0->getOffset();
        size_t d1_offset = d1->getOffset();
        size_t pS_offset = pS->getOffset();
        
        // swap d0 and pS
        Tree* t1 = new Tree(*tree);
        Node* t1_p  = t1->nodeWithOffset(p_offset);
        Node* t1_pA = t1->nodeWithOffset(pA_offset);
        Node* t1_pS = t1->nodeWithOffset(pS_offset);
        Node* t1_d0 = t1->nodeWithOffset(d0_offset);
        t1_p->removeDescendant(t1_d0);
        t1_pA->removeDescendant(t1_pS);
        t1_p->addDescendant(t1_pS);
        t1_pA->addDescendant(t1_d0);
        t1_d0->setAncestor(t1_pA);
        t1_pS->setAncestor(t1_p);
        t1->initializeDownPassSequence();
        t1->setDescriptor();

        // swap d1 and pS
        Tree* t2 = new Tree(*tree);
        Node* t2_p  = t2->nodeWithOffset(p_offset);
        Node* t2_pA = t2->nodeWithOffset(pA_offset);
        Node* t2_pS = t2->nodeWithOffset(pS_offset);
        Node* t2_d1 = t2->nodeWithOffset(d1_offset);
        t2_p->removeDescendant(t2_d1);
        t2_pA->removeDescendant(t2_pS);
        t2_p->addDescendant(t2_pS);
        t2_pA->addDescendant(t2_d1);
        t2_d1->setAncestor(t2_pA);
        t2_pS->setAncestor(t2_p);
        t2->initializeDownPassSequence();
        t2->setDescriptor();
                
        // add the information to neighbors
        neighbors[k++] = t1->getHash();
        neighbors[k++] = t2->getHash();
        
        // and update the tree list, possibly deleting the tree if its already in the list
        if (treeList->getTree(t1->getHash()) == nullptr)
            treeList->addTree(t1);
        else 
            delete t1;
        if (treeList->getTree(t2->getHash()) == nullptr)
            treeList->addTree(t2);
        else 
            delete t2;
        }
}

TreeNeighborhoodNni2::TreeNeighborhoodNni2(TreeList* tl) : TreeNeighborhoodNni(tl) {

}

TreeHashVec& TreeNeighborhoodNni2::getNeighbors(uint64_t treeHash) {

    TreeNeighbors::iterator it = nniNniNeighborhood.find(treeHash);
    if (it == nniNniNeighborhood.end())
        {
        std::vector<uint64_t> neighbors;
        generateNniNniNeighbors(treeHash, neighbors);
        nniNniNeighborhood.insert( std::make_pair(treeHash,neighbors) );
        it = nniNniNeighborhood.find(treeHash);
        return it->second;
        }
    return it->second;
}

TreeHashVec& TreeNeighborhoodNni2::getNeighbors(Tree* tree) {

    return TreeNeighborhoodNni2::getNeighbors(tree->getHash());
}

void TreeNeighborhoodNni2::generateNniNniNeighbors(uint64_t treeHash, std::vector<uint64_t>& neighbors) {

    TreeHashVec& neighbors1 = TreeNeighborhoodNni::getNeighbors(treeHash);
    std::unordered_set<uint64_t> uniqueNeighbors;
    for (uint64_t& x : neighbors1)
        {
        uniqueNeighbors.insert(x);
        TreeHashVec& neighbors2 = TreeNeighborhoodNni::getNeighbors(x);
        for (uint64_t& y : neighbors2)
            {
            if (y != treeHash)
                uniqueNeighbors.insert(y);
            }
        }
        
    neighbors.resize(uniqueNeighbors.size());
    size_t i = 0;
    for (const uint64_t& x : uniqueNeighbors)
        neighbors[i++] = x;
}
