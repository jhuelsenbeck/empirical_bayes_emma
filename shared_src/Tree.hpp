#ifndef Tree_hpp
#define Tree_hpp

#include <map>
#include <string>
#include <unordered_set>
#include <vector>
class CompactTree;
class Node;
class RandomVariable;
class TreeNeighborGenerator;
class TreeSpace;



class Tree {

    friend                          TreeSpace;
    friend                          TreeNeighborGenerator;
    
    public:
                                    Tree(void) = delete;
                                    Tree(const Tree& t);
                                    Tree(RandomVariable* rng, std::vector<std::string>& taxonNames);
                                    Tree(std::string newickStr, std::vector<std::string>& taxonNames, bool useDescriptor);
                                    Tree(CompactTree* newickBitSet, int nt);
                                    Tree(CompactTree* newickBitSet, std::vector<std::string>& taxonNames);
                                    Tree(Node* rootNode, std::vector<std::string>& taxonNames);
                                    Tree(Tree* t0, Tree* t1, Node* p0, Node* p1);
                                   ~Tree(void);
        Tree&                       operator=(const Tree& rhs);
        CompactTree*                getCompactRepresentation(void);
        uint64_t                    getHash(void) { return hash; }
        std::vector<Node*>&         getDownPassSequence(void) { return downPassSequence; }
        std::string                 getNewickString(void);
        int                         getNumNodes(void) { return numNodes; }
        int                         getNumTips(void) { return numTips; }
        Node*                       getRoot(void) { return root; }
        Node*                       findNodeWithIndex(int idx);
        Node*                       findTaxonNamed(std::string tName);
        bool                        hasNode(Node* p);
        void                        initializeDownPassSequence(void);
        void                        markNodesDownFromNode(Node* p);
        void                        markUpClsAsDirtyFromNode(Node* p);
        void                        markDnClsAsDirty(void);
        void                        markUpClsAsDirty(void);
        Node*                       nodeWithOffset(size_t idx) { return nodes[idx]; }
        int                         numTbrNeighbors(void);
        void                        print(void);
        void                        print(std::string header);
        void                        print(Node* subtree);
        void                        rerootOnNode(Node* rootNode);
        void                        rerootOnNode(Node* rootNode, std::vector<Node*>& nodeVec);
        void                        rerootOnTipZero(void);
        void                        setAllFlags(bool tf);
        void                        setDescriptor(void);
        std::pair<Tree*,Tree*>      split(Node* p);
    
    private:
        Node*                       addNode(void);
        Node*                       addNode(std::vector<Node*>& nodeVec);
        void                        clone(const Tree& t);
        Node*                       clone(const Tree& t, std::vector<Node*>& nodeVec);
        Node*                       cloneNodeStructure(Node* originalNode);
        void                        collapseSingletonInternalNode(Node* n);
        void                        debugPrint(std::string header);
        void                        deleteNodes(void);
        int                         indexForTaxonName(std::string& name, std::vector<std::string>& taxonNames);
        int                         numBits(int n);
        void                        passDown(Node* p);
        void                        removeNodesAbove(Node* newRoot);
        void                        removeNodesBelow(Node* newRoot);
        bool                        removeSuperfluousNodes(std::unordered_set<Node*>& removedNodes);
        void                        showNode(Node* p, int indent);
        std::vector<std::string>    tokenizeNewickString(std::string newickStr);
        void                        writeTree(Node* p, std::stringstream& strm);
        Node*                       root;
        int                         numNodes;
        int                         numTips;
        uint64_t                    hash;
        std::vector<Node*>          nodes;
        std::vector<Node*>          downPassSequence;
};

#endif
