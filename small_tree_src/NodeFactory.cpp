#include "Node.hpp"
#include "NodeFactory.hpp"



NodeFactory::NodeFactory(void) : chunkSize(4096) {

    Node* initialNodeVector = new Node[chunkSize]; // makes certain that most of the nodes are contiguous
    for (size_t i=0; i<chunkSize; i++)
        {
        Node* n = &initialNodeVector[i];
        allocated.push_back(n);
        pool.push_back(n);
        }
}

NodeFactory::~NodeFactory(void) {

    for (size_t i=0; i<allocated.size(); i += chunkSize)
        delete [] allocated[i];
}

Node* NodeFactory::getNode(void) {

    if (pool.empty() == true)
        {
        // if the node pool is empty, we allocate a new block of nodes
        Node* nodeVector = new Node[chunkSize];
        for (size_t i=0; i<chunkSize; i++)
            {
            allocated.push_back(&nodeVector[i]);
            pool.push_back(&nodeVector[i]);
            }
        }
    
    // return a node from the node pool, remembering to remove it from the pool.
    Node* n = pool.back();
    pool.pop_back();
    return n;
}

void NodeFactory::returnToPool(Node* n) {

    n->clean();
    pool.push_back( n );
}
