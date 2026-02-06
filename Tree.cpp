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

    // make an initial three-species tree
    root = addNode();
    root->setIndex(0);
    root->setName(taxonNames[0]);
    root->setIsTip(true);
    Node* rootDes = addNode();
    root->addDescendant(rootDes);
    rootDes->setAncestor(root);
    for (size_t i=1; i<3; i++)
        {
        Node* p = addNode();
        p->setIndex(static_cast<int>(i));
        p->setName(taxonNames[i]);
        p->setIsTip(true);
        p->setAncestor(rootDes);
        rootDes->addDescendant(p);
        }
        
    // randomly add the remaining taxa
    for (size_t i=3; i<taxonNames.size(); i++)
        {
        Node* p = nullptr;
        do {
            p = nodes[static_cast<size_t>(rng->uniformRv()*nodes.size())];
            } while (p == root);
        Node* pAnc = p->getAncestor();
        
        Node* newTip = addNode();
        newTip->setIndex(static_cast<int>(i));
        newTip->setName(taxonNames[i]);
        newTip->setIsTip(true);
        Node* newInt = addNode();
        
        pAnc->removeDescendant(p);
        newInt->addDescendant(p);
        newInt->addDescendant(newTip);
        pAnc->addDescendant(newInt);
        p->setAncestor(newInt);
        newTip->setAncestor(newInt);
        newInt->setAncestor(pAnc);
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

Tree::Tree(std::string newickStr, std::vector<std::string>& taxonNames) {

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
        else if (p->getBrlen() < 0.0)
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

void Tree::deleteNodes(void) {

    NodeFactory& nf = NodeFactory::nodeFactory();
    for (size_t i=0, n=nodes.size(); i<n; i++)
        nf.returnToPool(nodes[i]);
    nodes.clear();
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

std::string Tree::getNewickString(void) {

    std::stringstream strm;
    writeTree(root, strm);
    strm << ";";
    return strm.str();
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

    showNode(root, 0);
}

void Tree::print(Node* subtree) {

    showNode(subtree, 0);
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
    
    // if tip 0 is already the root, nothing to do
    if (tipZero == root)
        return;
            
    // store the path from tip 0 to the current root
    std::vector<Node*> pathToRoot;
    Node* current = tipZero;
    while (current != nullptr) 
        {
        pathToRoot.push_back(current);
        current = current->getAncestor();
        }
        
    // reverse the path direction, making tip 0 the new root
    for (size_t i = 0; i < pathToRoot.size() - 1; i++) 
        {
        Node* child = pathToRoot[i];
        Node* parent = pathToRoot[i + 1];
                
        // remove descendant from parent's descendant list
        parent->removeDescendant(child);
        
        // clear descendants's sibling pointer to avoid stale references
        child->setNextSibling(nullptr);
        
        // make parent a child of child (reverse the relationship)
        parent->setAncestor(child);
        child->addDescendant(parent);
        
        // handle branch lengths - the branch length stays with the node that's being moved
        if (i == 0) 
            {
            // first reversal: tip 0 becomes root, so it has no branch length
            parent->setBrlen(child->getBrlen());
            child->setBrlen(0.0); // Root has no branch length
            }
        }
    
    // set tipZero as the new root
    tipZero->setAncestor(nullptr);
    root = tipZero;
            
    // reinitialize the down pass sequence since tree topology changed
    initializeDownPassSequence();
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
        std::cout << std::endl;
        
        // recurse into descendants
        for (Node* d = p->getFirstDescendant(); d != nullptr; d = d->getNextSibling())
            showNode(d, indent + 3);
        }
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
    
    if (p->getIsTip() == false)
        strm << "(";
    else
        strm << p->getName() << ":" << p->getBrlen();
    
    // LCRS iteration over children
    bool first = true;
    for (Node* d = p->getFirstDescendant(); d != nullptr; d = d->getNextSibling())
        {
        if (!first)
            strm << ",";
        first = false;
        writeTree(d, strm);
        }
        
    if (p->getIsTip() == false)
        strm << ")" << ":" << p->getBrlen();
}

