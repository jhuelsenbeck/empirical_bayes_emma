#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <vector>
#include "BitSet.hpp"
#include "BitSetFactory.hpp"
#include "Msg.hpp"
#include "Node.hpp"
#include "NodeFactory.hpp"
#include "RandomVariable.hpp"
#include "Tree.hpp"
#define XXH_INLINE_ALL
#include "xxhash.h"




Tree::Tree(const Tree& t) {

    clone(t);
}

Tree::Tree(RandomVariable* rng, std::vector<std::string>& taxonNames) {

    numTips = static_cast<int>(taxonNames.size());
    numNodes = 0;

    if (numTips < 3)
        Msg::error("Must have at least three taxa");
        
    // build three species tree
    root = addNode();
    root->setIsTip(true);
    root->setIndex(0);
    root->setName(taxonNames[0]);
    
    Node* p = addNode();
    p->setAncestor(root);
    root->addDescendant(p);
    
    Node* q = addNode();
    q->setIsTip(true);
    q->setIndex(1);
    q->setName(taxonNames[1]);
    q->setAncestor(p);
    p->addDescendant(q);
    
    q = addNode();
    q->setIsTip(true);
    q->setIndex(2);
    q->setName(taxonNames[2]);
    q->setAncestor(p);
    p->addDescendant(q);
        
    for (int i=3; i<numTips; i++)
        {
        p = nullptr;
        do {
            p = nodes[(int)(rng->uniformRv()*nodes.size())];
            } while (p == root);
        Node* pAnc = p->getAncestor();
        
        Node* newTip = addNode();
        newTip->setIsTip(true);
        newTip->setIndex(i);
        newTip->setName(taxonNames[i]);
        
        Node* newInt = addNode();
        
        pAnc->removeDescendant(p);
        pAnc->addDescendant(newInt);
        newInt->setAncestor(pAnc);
        newInt->addDescendant(p);
        newInt->addDescendant(newTip);
        p->setAncestor(newInt);
        newTip->setAncestor(newInt);
        }

    // get postorder traversal sequence
    initializeDownPassSequence();
    
    // set the initial branch lengths
    for (Node* p : downPassSequence) 
        {
        if (p != root)
            p->setBrlen(0.02);
        }
        
    // relabel interior nodes
    int intIdx = numTips;
    for (Node* p : downPassSequence) 
        {
        if (p->getIsTip() == false)
            p->setIndex(intIdx++);
        }
        
    setDescriptor();
}

Tree::Tree(std::string newickStr, std::vector<std::string>& taxonNames, bool useDescriptor) {

    std::vector<std::string> tokens = tokenizeNewickString(newickStr);
    
    bool readingBrlen = false;
    Node* p = nullptr;
    int ntips = 0;
    numNodes = 0;
    for (size_t i=0; i<tokens.size(); i++)
        {
        std::string token = tokens[i];
        
        if (token == "(")
            {
            Node* newNode = addNode();
            newNode->setBrlen(-1.0);
            if (p == nullptr)
                {
                root = newNode;
                }
            else 
                {
                newNode->setAncestor(p);
                p->addDescendant(newNode);
                }
            p = newNode;
            }
        else if (token == ")" || token == ",")
            {
            if (p->getAncestor() == nullptr)
                Msg::error("Cannot move down tree when building Newick tree");
            p = p->getAncestor();
            }
        else if (token == ":")
            {
            readingBrlen = true;
            }
        else if (token == ";")
            {
            if (p != root)
                Msg::error("Expecting to end at the root of the tree");
            }
        else 
            {
            if (readingBrlen == false)
                {
                Node* newNode = addNode();
                newNode->setBrlen(-1.0);
                int idx = indexForTaxonName(token, taxonNames);
                if (idx == -1)
                    Msg::error("Could not find name in taxon names list");
                newNode->setIndex(idx);
                newNode->setName(token);
                newNode->setIsTip(true);
                newNode->setAncestor(p);
                p->addDescendant(newNode);
                p = newNode;
                ntips++;
                }
            else 
                {
                p->setBrlen(atof(token.c_str()));
                readingBrlen = false;
                }
            }
        }

    // get postorder traversal sequence
    initializeDownPassSequence();

    numTips = static_cast<int>(taxonNames.size());
    if (ntips != numTips)
        Msg::error("Mismatch in names during construction of Newick formatted tree");
        
    //rerootOnTipZero();

    // relabel interior nodes
    int intIdx = numTips;
    for (Node* p : downPassSequence)
        {
        if (p->getIsTip() == false)
            p->setIndex(intIdx++);
        }
        
    // set branch lengths
    for (Node* p : downPassSequence)
        {
        if (p == root)
            p->setBrlen(0.0);
        else if (p->getBrlen() < 0.0)
            p->setBrlen(0.02);
        }

    if (useDescriptor == true)
        setDescriptor();
}

Tree::Tree(BitSet* newickBitSet, std::vector<std::string>& taxonNames) {

    int numTaxonBits = numBits((int)taxonNames.size());
    Node* p = nullptr;
    int ntips = 0;
    numNodes = 0;

    bool stopTraversal = false;
    int i = 0;
    while (stopTraversal == false)
        {
        bool bit0 = (*newickBitSet)[i++];
        bool bit1 = (*newickBitSet)[i++];
        std::cout << bit0 << bit1 << std::endl;

        if (bit0 == false && bit1 == true) // 01 -> "("
            {
            Node* newNode = addNode();
            newNode->setBrlen(-1.0);
            if (p == nullptr)
                {
                root = newNode;
                }
            else 
                {
                newNode->setAncestor(p);
                p->addDescendant(newNode);
                }
            p = newNode;
            }
        else if (bit0 == true && bit1 == false) // 10 -> ")" or ","
            {
            if (p->getAncestor() == nullptr)
                Msg::error("Cannot move down tree when building Newick tree");
            p = p->getAncestor();
            }
        else if (bit0 == true && bit1 == true) // 11 -> ";"
            {
            stopTraversal = true;
            }
        else if (bit0 == false && bit1 == false) // 00 -> taxon number followed by index for taxon
            {
            int idx = 0;
            int power2 = pow(2.0,numTaxonBits-1);
            for (int j=0; j<numTaxonBits; j++, i++)
                {
                idx += power2 * (int)((*newickBitSet)[i]);
                power2 /= 2;
                }
            
            Node* newNode = addNode();
            newNode->setIndex(idx);
            newNode->setName(taxonNames[idx]);
            newNode->setIsTip(true);
            newNode->setAncestor(p);
            p->addDescendant(newNode);
            p = newNode;
            ntips++;
            }
        }

    // get postorder traversal sequence
    initializeDownPassSequence();

    numTips = static_cast<int>(taxonNames.size());
    if (ntips != numTips)
        Msg::error("Mismatch in names during construction of Newick formatted tree");
        
    rerootOnTipZero();

    // relabel interior nodes
    int intIdx = numTips;
    for (Node* p : downPassSequence)
        {
        if (p->getIsTip() == false)
            p->setIndex(intIdx++);
        }
        
    // set branch lengths
    for (Node* p : downPassSequence)
        {
        if (p == root)
            p->setBrlen(0.0);
        else
            p->setBrlen(0.02);
        }

    setDescriptor();
}

Tree::Tree(Node* rootNode, std::vector<std::string>& taxonNames) {

    numTips = static_cast<int>(taxonNames.size());
    numNodes = 0;
    
    // clone the node structure
    root = cloneNodeStructure(rootNode);
    
    // get postorder traversal sequence
    initializeDownPassSequence();
    
    // reroot the tree on tip zero
    rerootOnTipZero();
    
    // relabel interior nodes
    int intIdx = numTips;
    for (Node* p : downPassSequence)
        {
        if (p->getIsTip() == false)
            p->setIndex(intIdx++);
        }
        
    // set branch lengths
    for (Node* p : downPassSequence)
        {
        if (p == root)
            p->setBrlen(0.0);
        else
            p->setBrlen(0.02);
        }

    setDescriptor();
}

Tree::Tree(Tree* t0, Tree* t1, Node* p0, Node* p1) {

    // check the configuration
    if (t0->findNodeWithIndex(0) == nullptr)
        Msg::error("Expecting t0 to have the tip with index 0");
    if (t0->hasNode(p0) == false)
        Msg::error("Cannot find p0 in tree t0");
    if (t1->hasNode(p1) == false)
        Msg::error("Cannot find p1 in tree t1");
        
    // merge nodes into this tree
    std::vector<Node*> newNodes0, newNodes1;
    Node* root0 = clone(*t0, newNodes0);
    Node* root1 = clone(*t1, newNodes1);
    Node* c0 = newNodes0[p0->getOffset()];
    Node* c1 = newNodes1[p1->getOffset()];

    // add a node to the upper subtree if it has more than one node
    if (t0->numTips > 1 && t1->numTips > 1)
        {
        // insert a node in t1
        Node* newNode = addNode(newNodes1);
        Node* p = c1;
        Node* pAnc = p->getAncestor();
        pAnc->removeDescendant(p);
        pAnc->addDescendant(newNode);
        newNode->setAncestor(pAnc);
        newNode->addDescendant(p);
        p->setAncestor(newNode);
        newNode->setBrlen(p->getBrlen()*0.5);
        p->setBrlen(newNode->getBrlen());
        rerootOnNode(newNode, newNodes1);
        root1 = newNode;
        }
    
    // merge nodes
    for (Node* p : newNodes0)
        this->nodes.push_back(p);
    for (Node* p : newNodes1)
        this->nodes.push_back(p);
    for (int i=0; i<this->nodes.size(); i++)
        nodes[i]->setOffset(i);
    numNodes = (int)nodes.size();
    numTips = t0->getNumTips() + t1->getNumTips();
    
    // rearrange node pointers
    Node* newNode1 = addNode();
    if (t0->getNumNodes() == 1)
        {
        root = root1;
        Node* d = c0;
        Node* p = c1;
        Node* pAnc = p->getAncestor();
        pAnc->removeDescendant(p);
        pAnc->addDescendant(newNode1);
        newNode1->setAncestor(pAnc);
        newNode1->addDescendant(d);
        newNode1->addDescendant(p);
        p->setAncestor(newNode1);
        d->setAncestor(newNode1);
        rerootOnTipZero();
        }
    else if (t1->getNumNodes() == 1)
        {
        root = root0;
        Node* d = c1;
        Node* p = c0;
        Node* pAnc = p->getAncestor();
        pAnc->removeDescendant(p);
        pAnc->addDescendant(newNode1);
        newNode1->setAncestor(pAnc);
        newNode1->addDescendant(d);
        newNode1->addDescendant(p);
        p->setAncestor(newNode1);
        d->setAncestor(newNode1);
        initializeDownPassSequence();
        }
    else 
        {
        root = root0;
        Node* d = root1;
        Node* p = c0;
        Node* pAnc = p->getAncestor();
        pAnc->removeDescendant(p);
        pAnc->addDescendant(newNode1);
        newNode1->setAncestor(pAnc);
        newNode1->addDescendant(d);
        newNode1->addDescendant(p);
        p->setAncestor(newNode1);
        d->setAncestor(newNode1);
        initializeDownPassSequence();
        }
        
    // initialize down pass sequence
    initializeDownPassSequence();

    // relabel interior nodes
    int intIdx = numTips;
    for (Node* p : downPassSequence)
        {
        if (p->getIsTip() == false)
            p->setIndex(intIdx++);
        }
        
    // set branch lengths
    for (Node* p : downPassSequence)
        {
        if (p == root)
            p->setBrlen(0.0);
        else
            p->setBrlen(0.02);
        }

    setDescriptor();
}

Tree::~Tree(void) {

    // free memory here
    deleteNodes();
}

Tree& Tree::operator=(const Tree& rhs) {

    if (this != &rhs)
        clone(rhs);
    return *this;
}

Node* Tree::addNode(void) {

    NodeFactory& nf = NodeFactory::nodeFactory();
    Node* newNode = nf.getNode();
    newNode->setOffset((int)nodes.size());
    nodes.push_back(newNode);
    newNode->setIndex(numNodes);
    numNodes++;
    return newNode;
}

Node* Tree::addNode(std::vector<Node*>& nodeVec) {

    NodeFactory& nf = NodeFactory::nodeFactory();
    Node* newNode = nf.getNode();
    newNode->setOffset((int)nodeVec.size());
    nodeVec.push_back(newNode);
    return newNode;
}

void Tree::clone(const Tree& t) {

    // set up nodes for this tree
    if (this->numNodes != t.numNodes)
        {
        deleteNodes();
        for (int i=0; i<t.numNodes; i++)
            addNode();
        }
    this->numNodes = t.numNodes;
    this->numTips = t.numTips;
    this->hash = t.hash;
    
    // specify the root
    this->root = nodes[t.root->getOffset()];
    
    // deep copy of node information
    for (int i=0; i<this->numNodes; i++)
        {
        Node* p = nodes[i];
        Node* q = t.nodes[i];
        
        p->setIndex(q->getIndex());
        p->setIsTip(q->getIsTip());
        p->setBrlen(q->getBrlen());
        p->setName(q->getName());
        
        if (q->getAncestor() != nullptr)
            p->setAncestor( nodes[q->getAncestor()->getOffset()] );
        else
            p->setAncestor(nullptr);
        p->removeAllDescendants();
        for (Node* r = q->getFirstDescendant(); r != nullptr; r = r->getNextSibling())
            p->addDescendant( nodes[r->getOffset()] );
        }
        
    this->downPassSequence.resize(t.downPassSequence.size());
    for (int i=0,n=(int)t.downPassSequence.size(); i<n; i++)
        this->downPassSequence[i] = nodes[t.downPassSequence[i]->getOffset()];
}

Node* Tree::clone(const Tree& t, std::vector<Node*>& nodeVec) {

    if (nodeVec.size() > 0)
        Msg::error("Can only clone into an empty vector");
        
    // set up nodes for this tree
    for (int i=0; i<t.numNodes; i++)
        addNode(nodeVec);
    
    // specify the root
    Node* tempRoot = nodeVec[t.root->getOffset()];
    
    // deep copy of node information
    for (int i=0; i<t.numNodes; i++)
        {
        Node* p = nodeVec[i];
        Node* q = t.nodes[i];
        
        p->setIndex(q->getIndex());
        p->setIsTip(q->getIsTip());
        p->setBrlen(q->getBrlen());
        p->setName(q->getName());
        
        if (q->getAncestor() != nullptr)
            p->setAncestor( nodeVec[q->getAncestor()->getOffset()] );
        else
            p->setAncestor(nullptr);
        p->removeAllDescendants();
        for (Node* r = q->getFirstDescendant(); r != nullptr; r = r->getNextSibling())
            p->addDescendant( nodeVec[r->getOffset()] );
        }
        
    return tempRoot;
}

void Tree::collapseSingletonInternalNode(Node* n) {

    if (n == nullptr || n->getIsTip() == true)
        return;

    if (n->getNumDescendants() != 1)
        return;

    Node* anc = n->getAncestor();
    if (anc == nullptr)
        return;      // cannot collapse root

    Node* child = n->getFirstDescendant();
    if (child == nullptr)
        return;

    // remove n from its ancestor
    anc->removeDescendant(n);

    // attach the child directly to the ancestor
    anc->addDescendant(child);
    child->setAncestor(anc);
}

void Tree::deleteNodes(void) {

    NodeFactory& nf = NodeFactory::nodeFactory();
    for (size_t i=0, n=nodes.size(); i<n; i++)
        nf.returnToPool(nodes[i]);
    nodes.clear();
    numNodes = 0;
}

Node* Tree::cloneNodeStructure(Node* originalNode) {
    
    if (originalNode == nullptr)
        return nullptr;
    
    // create a new node for this tree
    Node* newNode = addNode();
    
    // copy the properties from the original node
    newNode->setIndex(originalNode->getIndex());
    newNode->setIsTip(originalNode->getIsTip());
    newNode->setBrlen(originalNode->getBrlen());
    newNode->setFlag(originalNode->getFlag());
    
    // copy the name
    if (originalNode->getName() != nullptr)
        newNode->setName(originalNode->getName());
    
    // recursively clone all descendants
    for (Node* child = originalNode->getFirstDescendant(); child != nullptr; child = child->getNextSibling()) 
        {
        Node* newChild = cloneNodeStructure(child);
        newChild->setAncestor(newNode);
        newNode->addDescendant(newChild);
        }   
    
    return newNode;
}

void Tree::debugPrint(std::string header) {

    std::cout << header << std::endl;
    for (size_t i=0; i<nodes.size(); i++) 
        {
        Node* p = nodes[i];
        std::cout << "   " << i << " -- ";
        std::cout << p << " ";
        std::cout << std::setw(3) << p->getIndex() << " offset = " << std::setw(3) << p->getOffset() << " ( ";
        if (p->getAncestor() != nullptr)
            std::cout << "a_" << p->getAncestor()->getIndex() << " ";
        else 
            std::cout << "a_NULL ";
        for (Node* d=p->getFirstDescendant(); d != nullptr; d = d->getNextSibling())
            std::cout << d->getIndex() << " ";
        std::cout << ")";
        if (p == root)
            std::cout << " <- Root";
        std::cout << std::endl;
        }
}

Node* Tree::findNodeWithIndex(int idx) {

    for (Node* p : downPassSequence)
        {
        if (p->getIndex() == idx)
            return p;
        }
    return nullptr;
}

Node* Tree::findTaxonNamed(std::string tName) {
    
    for (Node* p : downPassSequence)
        {
        if (p->getIsTip() == true)
            {
            if (p->getName() == tName)
                return p;
            }
        }
    return nullptr;
}

BitSet* Tree::getCompactRepresentation(void) {

    // 00 : next numTaxonBits are a taxon index
    // 01 : (
    // 10 : ) or ,
    // 11 : ;
    
    int numTaxonBits = numBits(numTips);
    int numCommas = numTips - 1;
    int numLeftParantheses = numTips - 2;
    int numRightParantheses = numTips - 2;
    size_t numBits = 2 * (numLeftParantheses + numRightParantheses + numCommas) + numTips * (2 + numTaxonBits) + 2;
    
    BitSet* treeBits = new BitSet(numBits);
    size_t pos = 0;
    writeTreeBits(root->getFirstDescendant(), treeBits, pos, numTaxonBits);
    treeBits->set(pos++);
    treeBits->set(pos++);
    
    return treeBits;
}

std::string Tree::getNewickString(void) {

    std::stringstream strm;
    writeTree(root->getFirstDescendant(), strm);
    strm << ");";
    return strm.str();
}

bool Tree::hasNode(Node* p) {

    for (int i=0; i<nodes.size(); i++)
        {
        if (nodes[i] == p)
            return true;
        }
    return false;
}

int Tree::indexForTaxonName(std::string& name, std::vector<std::string>& taxonNames) {

    for (size_t i=0; i<taxonNames.size(); i++)
        {
        if (name == taxonNames[i])
            return static_cast<int>(i);
        }
    return -1;
}

void Tree::initializeDownPassSequence(void) {

    downPassSequence.clear();
    passDown(root);
}

void Tree::markNodesDownFromNode(Node* p) {

    while (p != nullptr)
        {
        p->setFlag(true);
        p = p->getAncestor();
        }
}

void Tree::markUpClsAsDirtyFromNode(Node* p) {

    while (p != nullptr)
        {
        p->setDirtyUpCl(true);
        p = p->getAncestor();
        }
}

void Tree::markUpClsAsDirty(void) {

    for (Node* p : downPassSequence)
        {
        if (p->getIsTip() == false)
            p->setDirtyUpCl(true);
        else 
            p->setDirtyUpCl(false);
        }
}

void Tree::markDnClsAsDirty(void) {

    for (Node* p : downPassSequence)
        {
        if (p->getAncestor() != nullptr)
            {
            if (p->getAncestor() == root)
                p->setDirtyDnCl(false);
            else 
                p->setDirtyDnCl(true);
            }
        else 
            p->setDirtyDnCl(false);
        }
}

int Tree::numBits(int n) {

    int bits = 0;
    do 
        {
        ++bits;
        n >>= 1;
        } while (n);
    return bits;
}

int Tree::numTbrNeighbors(void) {

    int num = 0;
    for (Node* p : downPassSequence)
        {
        if (p->getAncestor() == nullptr)
            continue;
            
        if (p->getIsTip() == true)
            {
            p->scratchInt = 1;
            num += (2 * (numTips-1) - 3);
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
            num += n1 * n2;
            }
        }
        
    return num;
}

void Tree::passDown(Node* p) {

    if (p != nullptr)
        {
        // LCRS iteration over children
        for (Node* d = p->getFirstDescendant(); d != nullptr; d = d->getNextSibling())
            passDown(d);
        downPassSequence.push_back(p);
        }
}

void Tree::print(void) {

    std::cout << "   Number of tips  = " << numTips << std::endl;
    std::cout << "   Number of nodes = " << numNodes << " (" << nodes.size() << ")" << std::endl;
    showNode(root, 0);
}

void Tree::print(std::string header) {
    
    std::cout << "   " << header << std::endl;
    print();
}

void Tree::print(Node* subtree) {

    showNode(subtree, 0);
}

void Tree::removeNodesAbove(Node* newRoot) {

    // set the flag for all nodes above (and including) newRoot to true
    setAllFlags(false);
    newRoot->setFlag(true);
    for (std::vector<Node*>::reverse_iterator it=downPassSequence.rbegin(); it != downPassSequence.rend(); it++)
        {
        Node* anc = (*it)->getAncestor();
        if (anc == nullptr)
            continue;
        if (anc->getFlag() == true)
            (*it)->setFlag(true);
        }

    // add all nodes to be removed to a set
    std::unordered_set<Node*> nodesToDelete;
    for (Node* p : nodes)
        {
        if (p->getFlag() == true)
            nodesToDelete.insert(p);
        }
 
     // split the tree at that point
    newRoot->getAncestor()->removeDescendant(newRoot);
    newRoot->setAncestor(nullptr);
    newRoot->setNextSibling(nullptr);
    initializeDownPassSequence();

    if (removeSuperfluousNodes(nodesToDelete) == true)
        initializeDownPassSequence();
   
    // remove the nodes from the nodes vector of the tree
    std::unordered_set<Node*> removeSet(nodesToDelete.begin(), nodesToDelete.end());
    nodes.erase(
        std::remove_if(nodes.begin(), nodes.end(),
            [&](Node* p)
                {
                return removeSet.count(p) > 0;
                }),
        nodes.end());
    for (int i=0; i<nodes.size(); i++)
        nodes[i]->setOffset(i);
        
    // return the nodes to the factory
    NodeFactory& nf = NodeFactory::nodeFactory();
    for (Node* p : removeSet)
        nf.returnToPool(p);
        
    numNodes = (int)downPassSequence.size();
    numTips = 0;
    for (Node* p : downPassSequence)
        {
        if (p->getIsTip() == true)
            numTips++;
        }
}

void Tree::removeNodesBelow(Node* newRoot) {

    // set the flag for all nodes above (and including) newRoot to true
    setAllFlags(false);
    newRoot->setFlag(true);
    for (std::vector<Node*>::reverse_iterator it=downPassSequence.rbegin(); it != downPassSequence.rend(); it++)
        {
        Node* anc = (*it)->getAncestor();
        if (anc == nullptr)
            continue;
        if (anc->getFlag() == true)
            (*it)->setFlag(true);
        }
    
    // add all nodes to be removed to a set
    std::unordered_set<Node*> nodesToDelete;
    for (Node* p : nodes)
        {
        if (p->getFlag() == false)
            nodesToDelete.insert(p);
        }
        
    // split the tree at that point
    newRoot->getAncestor()->removeDescendant(newRoot);
    newRoot->setAncestor(nullptr);
    newRoot->setNextSibling(nullptr);
    root = newRoot;
    initializeDownPassSequence();
    
    // find tip with smallest index
    int smallestIdx = numNodes;
    for (Node* p : downPassSequence)
        {
        if (p->getIsTip() == false)
            continue;
        if (p->getIndex() < smallestIdx)
            smallestIdx = p->getIndex();
        }
    
    // reroot the tree on the tip with the smallest index
    rerootOnNode(findNodeWithIndex(smallestIdx));
    root->setBrlen(0.0);
    
    if (removeSuperfluousNodes(nodesToDelete) == true)
        initializeDownPassSequence();

    // remove the nodes from the nodes vector of the tree
    std::unordered_set<Node*> removeSet(nodesToDelete.begin(), nodesToDelete.end());
    nodes.erase(
        std::remove_if(nodes.begin(), nodes.end(),
            [&](Node* p)
                {
                return removeSet.count(p) > 0;
                }),
        nodes.end());
    for (int i=0; i<nodes.size(); i++)
        nodes[i]->setOffset(i);
        
    // return the nodes to the factory
    NodeFactory& nf = NodeFactory::nodeFactory();
    for (Node* p : removeSet)
        nf.returnToPool(p);

    numNodes = (int)downPassSequence.size();
    numTips = 0;
    for (Node* p : downPassSequence)
        {
        if (p->getIsTip() == true)
            numTips++;
        }
}

bool Tree::removeSuperfluousNodes(std::unordered_set<Node*>& removedNodes) {

    size_t nBefore = removedNodes.size();
    bool nodesRemoved = false;
    do {
        nodesRemoved = false;
        for (Node* p : downPassSequence)
            {
            if (p != root && p->getNumDescendants() == 1)
                {
                collapseSingletonInternalNode(p);
                removedNodes.insert(p);
                nodesRemoved = true;
                }
            }
        if (nodesRemoved == true)
            initializeDownPassSequence();
        } while (nodesRemoved == true);
        
    size_t nAfter = removedNodes.size();
        
    return (nBefore != nAfter);
}

void Tree::rerootOnNode(Node* rootNode) {

    if (rootNode == nullptr) 
        Msg::error("Cannot root on a null tip");
    
    // if tip 0 is already the root, nothing to do
    if (rootNode == root)
        return;
            
    // store the path from tip 0 to the current root
    std::vector<Node*> pathToRoot;
    Node* current = rootNode;
    while (current != nullptr) 
        {
        pathToRoot.push_back(current);
        current = current->getAncestor();
        }
        
    // reverse the path direction, making rootNode the new root
    for (size_t i=0; i<pathToRoot.size()-1; i++) 
        {
        Node* child = pathToRoot[i];
        Node* parent = pathToRoot[i + 1];
        Node* grandparent = parent->getAncestor(); // capture before any mutation
                
        // detach child from parent's descendant list
        parent->removeDescendant(child);
        
        // detach parent from grandparent's descendant list
        if (grandparent != nullptr)
            grandparent->removeDescendant(parent);
        
        // make parent a child of child (reverse the relationship)
        parent->setAncestor(child);
        child->addDescendant(parent);
        
        // handle branch lengths -- the branch length stays with the node that's being moved
        if (i == 0) 
            {
            // first reversal: rootNode becomes root, so it has no branch length
            parent->setBrlen(child->getBrlen());
            child->setBrlen(0.0); // Root has no branch length
            }
        }
    
    // set rootNode as the new root
    rootNode->setAncestor(nullptr);
    root = rootNode;
            
    // reinitialize the down pass sequence since tree topology changed
    initializeDownPassSequence();
}

void Tree::rerootOnNode(Node* rootNode, std::vector<Node*>& nodeVec) {

    if (rootNode->getAncestor() == nullptr) 
        return;
                
    // store the path from tip 0 to the current root
    std::vector<Node*> pathToRoot;
    Node* current = rootNode;
    while (current != nullptr) 
        {
        pathToRoot.push_back(current);
        current = current->getAncestor();
        }
        
    // reverse the path direction, making rootNode the new root
    for (size_t i=0; i<pathToRoot.size()-1; i++) 
        {
        Node* child = pathToRoot[i];
        Node* parent = pathToRoot[i + 1];
        Node* grandparent = parent->getAncestor(); // capture before any mutation
                
        // detach child from parent's descendant list
        parent->removeDescendant(child);
        
        // detach parent from grandparent's descendant list
        if (grandparent != nullptr)
            grandparent->removeDescendant(parent);
        
        // make parent a child of child (reverse the relationship)
        parent->setAncestor(child);
        child->addDescendant(parent);
        
        // handle branch lengths -- the branch length stays with the node that's being moved
        if (i == 0) 
            {
            // first reversal: rootNode becomes root, so it has no branch length
            parent->setBrlen(child->getBrlen());
            child->setBrlen(0.0); // Root has no branch length
            }
        }
    
    // set rootNode as the new root
    rootNode->setAncestor(nullptr);
}

void Tree::rerootOnTipZero(void) {

    // find the tip with index 0
    Node* tipZero = nullptr;
    for (Node* p : nodes) 
        {
        if (p->getIsTip() && p->getIndex() == 0) 
            {
            tipZero = p;
            break;
            }
        }
    if (tipZero == nullptr) 
        Msg::error("Could not find tip with index 0 for rerooting");
        
    rerootOnNode(tipZero);
}

void Tree::setAllFlags(bool tf) {

    for (Node* p : nodes)
        p->setFlag(tf);
}

void Tree::setDescriptor(void) {

    // get bit sets for each node
    BitSetFactory& bf = BitSetFactory::getFactory();
    std::vector<BitSet*> partitions(numNodes);
    for (size_t i=0; i<numNodes; i++)
        partitions[i] = bf.getBitSet();

    // initialize the bit set for the tip nodes
    for (size_t i=0; i<numTips; i++)
        partitions[i]->set(i);
        
    std::set<BitSet*,CompBitSet> nonTrivialPartitions;
    for (Node* p : downPassSequence)
        {
        if (p->getIsTip() == true)
            continue;
        if (p->getAncestor() == root)
            continue;
            
        for (Node* d=p->getFirstDescendant(); d != nullptr; d=d->getNextSibling())
            *(partitions[p->getIndex()]) |= *(partitions[d->getIndex()]);
        nonTrivialPartitions.insert(partitions[p->getIndex()]);
        }
                
    BitSet descriptor(nonTrivialPartitions.size() * numTips);
    int k = 0;
    for (std::set<BitSet*,CompBitSet>::iterator it = nonTrivialPartitions.begin(); it != nonTrivialPartitions.end(); it++)
        {
        for (int i=0; i<numTips; i++)
            {
            if ((*it)->isSet(i) == true)
                descriptor.set(k);
            k++;
            }
        }
    
    const void* dataPtr = static_cast<const void*>(descriptor.data().data());
    size_t byteLen = static_cast<size_t>(descriptor.data().size()) * sizeof(unsigned);
    hash = XXH3_64bits(dataPtr, byteLen);
    //std::cout << descriptor << std::endl;
    //std::cout << hash << std::endl;

    for (size_t i=0; i<partitions.size(); i++)
        bf.returnToPool(partitions[i]);
}

void Tree::showNode(Node* p, int indent) {

    if (p != nullptr)
        {
        std::cout << "   ";
        for (int i=0; i<indent; i++)
            std::cout << " ";
        std::cout << p->getIndex();
        if (p->getAncestor() != nullptr)
            std::cout << " ( a_" << p->getAncestor()->getIndex() << " ";
        else 
            std::cout << " ( a_null ";
        // iterate over descendants
        for (Node* d = p->getFirstDescendant(); d != nullptr; d = d->getNextSibling())
            std::cout << d->getIndex() << " ";
        std::cout << ") ";
        std::cout << std::fixed << std::setprecision(8) << p->getBrlen() << " ";
        std::cout << p->getDirtyDnCl() << p->getDirtyUpCl() << " ";
        std::cout << p->getName() << " ";
        if (p == root)
            std::cout << "<- Root ";
        std::cout << std::endl;
        
        // recurse into descendants
        for (Node* d = p->getFirstDescendant(); d != nullptr; d = d->getNextSibling())
            showNode(d, indent + 3);
        }
}

std::pair<Tree*,Tree*> Tree::split(Node* p) {

    Tree* t1 = new Tree(*this);
    Tree* t2 = new Tree(*this);
    
    t1->removeNodesAbove(t1->findNodeWithIndex(p->getIndex()));
    t2->removeNodesBelow(t2->findNodeWithIndex(p->getIndex()));
    
    std::pair<Tree*,Tree*> subtrees;
    subtrees.first = t1;
    subtrees.second = t2;
    return subtrees;
}

std::vector<std::string> Tree::tokenizeNewickString(std::string newickStr) {

    std::vector<std::string> tokens;
    
    std::string token = "";
    for (size_t i=0; i<newickStr.size(); i++)
        {
        char c = newickStr[i];
        
        if (c == '(' || c == ')' || c == ',' || c == ';' || c == ':')
            {
            if (token != "")
                {
                tokens.push_back(token);
                token = "";
                }
            tokens.push_back(std::string(1,c));
            }
        else 
            {
            token += std::string(1,c);
            }
        }
    if (token != "")
        {
        tokens.push_back(token);
        token = "";
        }
    
    return tokens;
}

void Tree::writeTree(Node* p, std::stringstream& strm) {

    if (p == nullptr)
        return;
    
    // special handling for root node
    if (p == root->getFirstDescendant())
        {        
        // three-way split
        strm << "(";
        bool first = true;
        for (Node* d = p->getFirstDescendant(); d != nullptr; d = d->getNextSibling())
            {
            if (!first)
                strm << ",";
            first = false;
            writeTree(d, strm);
            }
        strm << "," << root->getName();
        }
    else if (p->getIsTip() == false)
        {
        // internal node
        strm << "(";
        
        bool first = true;
        for (Node* d = p->getFirstDescendant(); d != nullptr; d = d->getNextSibling())
            {
            if (!first)
                strm << ",";
            first = false;
            writeTree(d, strm);
            }
            
        strm << ")";
        }
    else
        {
        // tip node
        strm << p->getName();
        }
}

void Tree::writeTreeBits(Node* p, BitSet* bitSet, size_t& pos, int numTaxonBits) {

    if (p == nullptr)
        return;
    
    // special handling for root node
    if (p == root->getFirstDescendant())
        {        
        // three-way split
        pos++;              // 0
        bitSet->set(pos++); // 1
        bool first = true;
        for (Node* d = p->getFirstDescendant(); d != nullptr; d = d->getNextSibling())
            {
            if (!first)
                {
                bitSet->set(pos++); // 1
                pos++;              // 0
                }
            first = false;
            writeTreeBits(d, bitSet, pos, numTaxonBits);
            }
        bitSet->set(pos++); // 1
        pos++;              // 0
        pos++;              // 0
        pos++;              // 0
        uint32_t n = root->getIndex();
        for (size_t i=0; i<numTaxonBits; i++, pos++) 
            {
            if (n & (1ULL << (numTaxonBits - 1 - i)))
                bitSet->set(pos);
            }
        }
    else if (p->getIsTip() == false)
        {
        // internal node
        pos++;              // 0
        bitSet->set(pos++); // 1
        
        bool first = true;
        for (Node* d = p->getFirstDescendant(); d != nullptr; d = d->getNextSibling())
            {
            if (!first)
                {
                bitSet->set(pos++); // 1
                pos++;              // 0
                }
            first = false;
            writeTreeBits(d, bitSet, pos, numTaxonBits);
            }
            
        bitSet->set(pos++); // 1
        pos++;              // 0
        }
    else
        {
        // tip node
        pos++;
        pos++;
        uint32_t n = p->getIndex();
        for (size_t i=0; i<numTaxonBits; i++, pos++) 
            {
            if (n & (1ULL << (numTaxonBits - 1 - i)))
                bitSet->set(pos);
            }
        }
}
