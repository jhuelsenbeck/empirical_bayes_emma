#include "Msg.hpp"
#include "Node.hpp"
#include "Tree.hpp"
#include "TreeNeighborGenerator.hpp"


TreeNeighborGenerator::TreeNeighborGenerator(TreeCache* tc) : treeCache(tc) {

}

bool TreeNeighborGenerator::checkTree(Tree* t) {

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

TreeNeighborGeneratorNNI::TreeNeighborGeneratorNNI(TreeCache* tc) : TreeNeighborGenerator(tc) {

}

void TreeNeighborGeneratorNNI::generateNeighbors(Tree* tree, std::vector<TreeInfo*>& neighbors) {

    // generate NNI neighbors
    size_t numNeighbors = 2 * (tree->getNumTips() - 3);
    neighbors.reserve(numNeighbors);
    
    const std::vector<Node*>& postOrder = tree->getDownPassSequence();
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
        TreeInfo* ti1 = getOrCreateTreeInfo(treeCache, t1);
        TreeInfo* ti2 = getOrCreateTreeInfo(treeCache, t2);
        neighbors.push_back(ti1);
        neighbors.push_back(ti2);
        if (ti1->hasLnLikelihood == false)
            ti1->tree = t1;
        else 
            delete t1;
        if (ti2->hasLnLikelihood == false)
            ti2->tree = t2;
        else 
            delete t2;
        }
}

void TreeNeighborGeneratorNNI::generateNeighbors(Tree* tree, std::vector<TreeInfo*>& neighbors, RandomVariable*) {

    generateNeighbors(tree, neighbors);
}

TreeNeighborGeneratorTBR::TreeNeighborGeneratorTBR(TreeCache* tc) : TreeNeighborGenerator(tc) {

}

void TreeNeighborGeneratorTBR::generateNeighbors(Tree* tree, std::vector<TreeInfo*>& neighbors) {

    // generate TBR neighbors
    int nNeighbors = tree->numTbrNeighbors();
    neighbors.reserve(nNeighbors);

    const std::vector<Node*>& postOrder = tree->getDownPassSequence();
    Node* rootNode = tree->getRoot();
    
    for (Node* p : postOrder)
        {
        if (p == rootNode)
            continue;
            
        // split the tree into two subtrees at p
        std::pair<Tree*,Tree*> subtrees = tree->split(p);
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
                if (t->getNumTips() != tree->getNumTips())
                    Msg::error("Trees are of different sizes");
                if (t->getNumNodes() != tree->getNumNodes())
                    Msg::error("Trees have different numbers of nodes");

                TreeInfo* ti = getOrCreateTreeInfo(treeCache, t);
                neighbors.push_back(ti);
                if (ti->hasLnLikelihood == false)
                    ti->tree = t;
                else 
                    delete t;
                }
            }
        
        delete t0;
        delete t1;
        }
        
    // check results
    if (nNeighbors != neighbors.size())
        Msg::error("Did not generate the expected number of TBR neighbors");

}

void TreeNeighborGeneratorTBR::generateNeighbors(Tree* tree, std::vector<TreeInfo*>& neighbors, RandomVariable*) {

    generateNeighbors(tree, neighbors);
}

void TreeNeighborGeneratorRandomTBR::generateNeighbors(Tree* tree, std::vector<TreeInfo*>& neighbors, RandomVariable* rng) {

}

void TreeNeighborGeneratorRandomTBR::generateNeighbors(Tree* tree, std::vector<TreeInfo*>& neighbors) {

    Msg::error("Neighbor generation under random TBR requires a random number generator");
}
