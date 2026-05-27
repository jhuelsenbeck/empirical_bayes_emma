#include <regex>
#include "Node.hpp"



Node::Node(void) : 
    ancestor(nullptr), firstDescendant(nullptr), nextSibling(nullptr), 
    index(0), brlen(0.0), offset(0), isTip(false), flag(false) {

    name[0] = '\0';
}

Node::~Node(void) {

}

void Node::addDescendant(Node* p) {
    
    if (firstDescendant == nullptr) 
        {
        // no descendants yet - this becomes the first descendant
        firstDescendant = p;
        } 
    else 
        {
        // find the last sibling and append
        Node* lastDescendant = firstDescendant;
        while (lastDescendant->nextSibling != nullptr)
            lastDescendant = lastDescendant->nextSibling;
        lastDescendant->nextSibling = p;
        }
    p->nextSibling = nullptr;  // ensure new descendant has no sibling
}

void Node::clean(void) {

    ancestor = nullptr;
    firstDescendant = nullptr;
    nextSibling = nullptr;
    brlen = 0.0;
    offset = 0;
    index = 0;
    name[0] = '\0';
    isTip = false;
    dirtyUpCl = true;
    dirtyDnCl = true;
}

Node* Node::getSister(void) {

    if (ancestor == nullptr)
        return nullptr;
    for (Node* p = ancestor->firstDescendant; p != nullptr; p = p->nextSibling)
        {
        if (p != this)
            return p;
        }
    return nullptr;
}

void Node::removeDescendant(Node* p) {
    
    if (firstDescendant == nullptr)
        return;
    
    if (firstDescendant == p) 
        {
        // removing the first child
        firstDescendant = p->nextSibling;
        p->nextSibling = nullptr;
        return;
        }
    
    // search through siblings to find predecessor
    Node* prev = firstDescendant;
    while (prev->nextSibling != nullptr && prev->nextSibling != p)
        prev = prev->nextSibling;
    if (prev->nextSibling == p) 
        {
        prev->nextSibling = p->nextSibling;
        p->nextSibling = nullptr;
        }
}

void Node::removeAllDescendants(void) {
    
    // clear sibling links for all children
    Node* descendant = firstDescendant;
    while (descendant != nullptr) 
        {
        Node* next = descendant->nextSibling;
        descendant->nextSibling = nullptr;
        descendant = next;
        }
    firstDescendant = nullptr;
}

int Node::getNumDescendants(void) {
    
    int count = 0;
    for (Node* d = firstDescendant; d != nullptr; d = d->nextSibling)
        count++;
    return count;
}

void Node::setName(char* s) {

    strcpy(name, s);
}

void Node::setName(std::string s) {

    strcpy(name, s.c_str());
}
