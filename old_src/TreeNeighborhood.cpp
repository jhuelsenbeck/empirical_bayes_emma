#include <algorithm>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include "Msg.hpp"
#include "Node.hpp"
#include "RandomVariable.hpp"
#include "Tree.hpp"
#include "TreeList.hpp"
#include "TreeNeighborhood.hpp"



TreeNeighborhood::TreeNeighborhood(TreeList* tl) : treeList(tl) {

}

TreeNeighborhood::~TreeNeighborhood(void) {

}

TreeNeighborhoodNni::TreeNeighborhoodNni(TreeList* tl) : TreeNeighborhood(tl) {

}

void TreeNeighborhood::getNeighbors(Tree* tree, NeighborValues& neighbors) {

    getNeighbors(tree->getHash(), neighbors);
}

void TreeNeighborhood::getNeighbors(uint64_t treeHash, TreeHashVec& vec) {

}

void TreeNeighborhood::getNeighbors(uint64_t treeHash, NeighborValues& neighbors, uint64_t defaultTree) {

    getNeighbors(treeHash, neighbors);
}

void TreeNeighborhood::getNeighbors(uint64_t treeHash, NeighborValues& neighbors) {

    TreeNeighbors::iterator it = neighborhood.find(treeHash);
    if (it == neighborhood.end())
        {
        generateNeighbors(treeHash, neighbors);
        }
    else 
        {
        neighbors.reserve(it->second.size());
        for (uint64_t& val : it->second)
            {
            if (treeList->isTreeInList(val) == false)
                {
                Tree* tree = treeList->getTree(val);
                neighbors.emplace_back(val, tree, 0.0);
                }
            else
                {
                TreeInfo& tInfo = treeList->getTreeInfo(val);
                neighbors.emplace_back(val, nullptr, tInfo.lnL);
                }

            }
        }
}

void TreeNeighborhoodNni::generateNeighbors(uint64_t treeHash, NeighborValues& neighbors) {

    Tree* tree = treeList->getTree(treeHash);
    if (tree == nullptr)
        Msg::error("Could not find tree in tree list");
    generateNeighbors(tree, neighbors);
}

void TreeNeighborhoodNni::generateNeighbors(Tree* tree, NeighborValues& neighbors) {

    // generate NNI neighbors
    uint64_t treeHash = tree->getHash();
        
    size_t numNeighbors = 2 * (tree->getNumTips() - 3);
    neighbors.reserve(numNeighbors);
    const std::vector<Node*>& postOrder = tree->getDownPassSequence();
    TreeHashVec& vec = neighborhood[treeHash];
    vec.reserve(numNeighbors);
    
    Node* treeRoot = tree->getRoot();
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
                
        // add the information to neighbors tuple vector and neighborhood map
        vec.push_back(t1->getHash());
        vec.push_back(t2->getHash());
        
        if (treeList->isTreeInList(t1->getHash()) == false)
            {
            neighbors.emplace_back(t1->getHash(), t1, 0.0);
            }
        else
            {
            TreeInfo& tInfo1 = treeList->getTreeInfo(t1->getHash());
            neighbors.emplace_back(t1->getHash(), nullptr, tInfo1.lnL);
            delete t1;
            }
        if (treeList->isTreeInList(t2->getHash()) == false)
            {
            neighbors.emplace_back(t2->getHash(), t2, 0.0);
            }
        else
            {
            TreeInfo& tInfo2 = treeList->getTreeInfo(t2->getHash());
            neighbors.emplace_back(t2->getHash(), nullptr, tInfo2.lnL);
            delete t2;
            }
            
        }
}

TreeNeighborhoodNni2::TreeNeighborhoodNni2(TreeList* tl) : TreeNeighborhoodNni(tl) {

}

void TreeNeighborhoodNni2::generateNeighbors(uint64_t treeHash, NeighborValues& neighbors) {
//
//    // generate NNI^2 neighbors
//    TreeNeighbors::iterator it = neighborhood.find(treeHash);
//    
//    TreeHashVec& neighbors1 = TreeNeighborhoodNni::getNeighbors(treeHash);
//    std::unordered_set<uint64_t> uniqueNeighbors;
//    for (uint64_t& x : neighbors1)
//        {
//        uniqueNeighbors.insert(x);
//        TreeHashVec& neighbors2 = TreeNeighborhoodNni::getNeighbors(x);
//        for (uint64_t& y : neighbors2)
//            {
//            if (y != treeHash)
//                uniqueNeighbors.insert(y);
//            }
//        }
//        
//    neighbors.resize(uniqueNeighbors.size());
//    size_t i = 0;
//    for (const uint64_t& x : uniqueNeighbors)
//        neighbors[i++] = x;
}

TreeNeighborhoodTbr::TreeNeighborhoodTbr(TreeList* tl) : TreeNeighborhood(tl) {

}

void TreeNeighborhoodTbr::generateNeighbors(Tree* tree, NeighborValues& neighbors) {

    getNeighbors(tree->getHash(), neighbors);
}

void TreeNeighborhoodTbr::generateNeighbors(uint64_t treeHash, NeighborValues& neighbors) {

    // generate TBR neighbors
    Tree* original = treeList->getTree(treeHash);
    if (original == nullptr)
        Msg::error("Could not find tree in tree list for generateTbrNeighbors");

    int nNeighbors = original->numTbrNeighbors();
    TreeHashVec& vec = neighborhood[treeHash];
    vec.reserve(nNeighbors);

    const std::vector<Node*>& postOrder = original->getDownPassSequence();
    Node* rootNode = original->getRoot();
    
    for (Node* p : postOrder)
        {
        if (p == rootNode)
            continue;
            
        // split the tree into two subtrees at p
        std::pair<Tree*,Tree*> subtrees = original->split(p);
        Tree* t0 = subtrees.first;
        Tree* t1 = subtrees.second;
        const std::vector<Node*>& postOrder0 = t0->getDownPassSequence();
        const std::vector<Node*>& postOrder1 = t1->getDownPassSequence();
        
        // generate trees from all combinations of branches in the two
        // subtrees that can be connected
        for (Node* p0 : postOrder0)
            {
            if (p0->getAncestor() == nullptr && t0->getNumNodes() > 1)
                continue;
                
            for (Node* p1 : postOrder1)
                {
                if (p1->getAncestor() == nullptr && t1->getNumNodes() > 1)
                    continue;
                    
                Tree* t = new Tree(t0, t1, p0, p1);
                //t->print();
                
                if (checkTree(t) == false)
                    Msg::error("Problem with TBR tree");
                if (t->getNumTips() != original->getNumTips())
                    Msg::error("Trees are of different sizes");
                if (t->getNumNodes() != original->getNumNodes())
                    Msg::error("Trees have different numbers of nodes");

                vec.push_back(t->getHash());
                
                if (treeList->isTreeInList(t->getHash()) == false)
                    {
                    neighbors.emplace_back(t->getHash(), t, 0.0);
                    }
                else
                    {
                    TreeInfo& tInfo = treeList->getTreeInfo(t->getHash());
                    neighbors.emplace_back(t->getHash(), nullptr, tInfo.lnL);
                    delete t;
                    }
                }
            }
        
        delete t0;
        delete t1;
        }
        
    // check results
    if (nNeighbors != neighbors.size())
        Msg::error("Did not generate the expected number of TBR neighbors");
        
#   if 0
    std::cout << "Expected number of TBR neighbors: " << original->numTbrNeighbors() << std::endl;
    std::cout << "Neighbors of tree " << treeHash << " (" << neighbors.size() << "): ";
    for (uint64_t h : neighbors)
        std::cout << h << " ";
    std::cout << std::endl;
    exit(1);
#   endif
}

bool TreeNeighborhoodTbr::checkTree(Tree* t) {

    bool isTreeGood = true;

    if (t->getRoot()->getIndex() != 0)
        isTreeGood = false;
    if (t->getRoot()->getNumDescendants() != 1)
        isTreeGood = false;
    int nTips = 0;
    for (Node* p : t->nodes)
        {
        if (p->getIsTip() == true)
            {
            nTips++;
            if (p->getNumDescendants() > 0 && p != t->getRoot())
                isTreeGood = false;
            if (p->getIndex() < 0 || p->getIndex() > t->getNumTips())
                isTreeGood = false;
            }
            
        
        }
    if (nTips != t->getNumTips())
        isTreeGood = false;
        
    if (t->nodes.size() != 2 * nTips - 2)
        isTreeGood = false;

    for (Node* p : t->nodes)
        {
        if (p->getIsTip() == true)
            {
            if (p->getIndex() < 0 || p->getIndex() > t->getNumTips())
                isTreeGood = false;
            }
        else 
            {
            if (p->getIndex() < nTips || p->getIndex() >= 2*nTips-2)
                isTreeGood = false;
            }
        }
    
    return isTreeGood;
}

TreeNeighborhoodTbrRandom::TreeNeighborhoodTbrRandom(TreeList* tl, int nn, RandomVariable* r) : 
    TreeNeighborhood(tl), numNeighbors(nn), rng(r) {

}

void TreeNeighborhoodTbrRandom::generateNeighbors(Tree* tree, NeighborValues& neighbors) {

    getNeighbors(tree->getHash(), neighbors);
}

void TreeNeighborhoodTbrRandom::generateNeighbors(uint64_t treeHash, NeighborValues& neighbors) {

    Tree* original = treeList->getTree(treeHash);
    if (original == nullptr)
        Msg::error("Could not find tree in tree list for generateNeighbors");

    const std::vector<Node*>& postOrder = original->getDownPassSequence();
    Node* rootNode = original->getRoot();
    int numTips = original->getNumTips();
    
    // pass 1: count valid reconnections N(b) for every candidate bisection
    // branch b. We do a real split so the counts match exactly what the
    // enumeration in pass 2 will produce.
#   if 0
    std::vector<size_t> bisectionOffsets;
    std::vector<size_t> cumNeighbors;
    bisectionOffsets.reserve(original->getNumNodes());
    cumNeighbors.reserve(original->getNumNodes());
    size_t total = 0;
    for (Node* p : postOrder)
        {
        if (p == rootNode)
            continue;
            
        std::pair<Tree*,Tree*> sub = original->split(p);
        Tree* t0 = sub.first;
        Tree* t1 = sub.second;
        size_t n0 = (t0->getNumNodes() == 1) ? 1 : (t0->getNumNodes() - 1);
        size_t n1 = (t1->getNumNodes() == 1) ? 1 : (t1->getNumNodes() - 1);
        size_t n  = n0 * n1;
        delete t0;
        delete t1;
        
        if (n == 0)
            continue;
        bisectionOffsets.push_back(p->getOffset());
        total += n;
        cumNeighbors.push_back(total);
        }
#   else
    std::vector<size_t> bisectionOffsets;
    std::vector<size_t> cumNeighbors;
    bisectionOffsets.reserve(original->getNumNodes());
    cumNeighbors.reserve(original->getNumNodes());
    size_t total = 0;
    for (Node* p : postOrder)
        {
        if (p == rootNode)
            continue;
            
        if (p->getIsTip() == true)
            {
            p->scratchInt = 1;
            total += (2 * (numTips-1) - 3);
            }
        else
            {
            int nTips1 = 0;
            for (Node* d=p->getFirstDescendant(); d != nullptr; d = d->getNextSibling())
                nTips1 += d->scratchInt;
            p->scratchInt = nTips1;
            int nTips2 = numTips - nTips1;
            int n1 = 1;
            if (nTips1 > 2)
                n1 = 2 * nTips1 - 3;
            int n2 = 1;
            if (nTips2 > 2)
                n2 = 2 * nTips2 - 3;
            total += n1 * n2;
            }

        bisectionOffsets.push_back(p->getOffset());
        cumNeighbors.push_back(total);
        }
#   endif   
     
    if (total != (size_t)original->numTbrNeighbors())
        Msg::error("TBR neighbor count does not match analytic value");
    
    // draw k distinct indices in [0, total) using Floyd's algorithm. Each
    // neighbor is selected with probability k/total uniformly across moves.
    size_t k = (size_t)numNeighbors;
    if (k > total)
        k = total;
    if (neighbors.size() == 1)  
        k--;
    std::unordered_set<size_t> picksSet;
    picksSet.reserve(k);
    for (size_t j=total-k; j<total; j++)
        {
        size_t r = (size_t)(rng->uniformRv() * (double)(j + 1));
        if (picksSet.insert(r).second == false)
            picksSet.insert(j);
        }
    std::vector<size_t> picks(picksSet.begin(), picksSet.end());
    std::sort(picks.begin(), picks.end());
    
    // pass 2: walk the sorted picks. Each chosen bisection branch is split
    // at most once and reused across all of its sampled local indices.
    neighbors.reserve(k);
    size_t bIdx     = 0;
    size_t prevBIdx = SIZE_MAX;
    Tree*  t0       = nullptr;
    Tree*  t1       = nullptr;
    for (size_t globalIdx : picks)
        {
        while (cumNeighbors[bIdx] <= globalIdx)
            bIdx++;
        size_t localIdx = globalIdx - (bIdx == 0 ? 0 : cumNeighbors[bIdx-1]);
        
        if (bIdx != prevBIdx)
            {
            delete t0;
            delete t1;
            Node* pBisect = original->nodeWithOffset(bisectionOffsets[bIdx]);
            std::pair<Tree*,Tree*> sub = original->split(pBisect);
            t0 = sub.first;
            t1 = sub.second;
            prevBIdx = bIdx;
            }
        
        // the downpass visits the subtree root last, so skipping the root
        // (when numNodes > 1) is equivalent to clamping the valid range to
        // [0, nValid), and postOrder[i] gives the desired node directly
        const std::vector<Node*>& postOrder0 = t0->getDownPassSequence();
        const std::vector<Node*>& postOrder1 = t1->getDownPassSequence();
        size_t n1Valid = (t1->getNumNodes() == 1) ? 1 : (postOrder1.size() - 1);
        size_t i0 = localIdx / n1Valid;
        size_t i1 = localIdx % n1Valid;
        Node* p0 = postOrder0[i0];
        Node* p1 = postOrder1[i1];
        
        Tree* t = new Tree(t0, t1, p0, p1);
        if (checkTree(t) == false)
            Msg::error("Problem with TBR tree");
        if (t->getNumTips() != original->getNumTips())
            Msg::error("Trees are of different sizes");
        if (t->getNumNodes() != original->getNumNodes())
            Msg::error("Trees have different numbers of nodes");
        
        if (treeList->isTreeInList(t->getHash()) == false)
            {
            neighbors.emplace_back(t->getHash(), t, 0.0);
            }
        else
            {
            TreeInfo& tInfo = treeList->getTreeInfo(t->getHash());
            neighbors.emplace_back(t->getHash(), nullptr, tInfo.lnL);
            delete t;
            }
        }
    delete t0;
    delete t1;
}

void TreeNeighborhoodTbrRandom::getNeighbors(uint64_t treeHash, NeighborValues& neighbors, uint64_t defaultTree) {

    if (treeList->isTreeInList(defaultTree) == false)
        {
        Tree* tree = treeList->getTree(defaultTree);
        neighbors.emplace_back(defaultTree, tree, 0.0);
        }
    else
        {
        TreeInfo& tInfo = treeList->getTreeInfo(defaultTree);
        neighbors.emplace_back(defaultTree, nullptr, tInfo.lnL);
        }

    TreeNeighborhood::getNeighbors(treeHash, neighbors);
}

bool TreeNeighborhoodTbrRandom::checkTree(Tree* t) {

    bool isTreeGood = true;

    if (t->getRoot()->getIndex() != 0)
        isTreeGood = false;
    if (t->getRoot()->getNumDescendants() != 1)
        isTreeGood = false;
    int nTips = 0;
    for (Node* p : t->nodes)
        {
        if (p->getIsTip() == true)
            {
            nTips++;
            if (p->getNumDescendants() > 0 && p != t->getRoot())
                isTreeGood = false;
            if (p->getIndex() < 0 || p->getIndex() > t->getNumTips())
                isTreeGood = false;
            }
            
        
        }
    if (nTips != t->getNumTips())
        isTreeGood = false;
        
    if (t->nodes.size() != 2 * nTips - 2)
        isTreeGood = false;

    for (Node* p : t->nodes)
        {
        if (p->getIsTip() == true)
            {
            if (p->getIndex() < 0 || p->getIndex() > t->getNumTips())
                isTreeGood = false;
            }
        else 
            {
            if (p->getIndex() < nTips || p->getIndex() >= 2*nTips-2)
                isTreeGood = false;
            }
        }
    
    return isTreeGood;
}

