#ifndef Tree_hpp
#define Tree_hpp

#include <map>
#include <string>
#include <vector>
class BitSet;
class Node;
class RandomVariable;
class TreeSpace;



class Tree {

    friend                          TreeSpace;
    
    public:
                                    Tree(void) = delete;
                                    Tree(const Tree& t);
                                    Tree(RandomVariable* rng, std::vector<std::string>& taxonNames);
                                    Tree(std::string newickStr, std::vector<std::string>& taxonNames);
                                   ~Tree(void);
        Tree&                       operator=(const Tree& rhs);
        uint64_t                    getHash(void) { return hash; }
        std::vector<Node*>&         getDownPassSequence(void) { return downPassSequence; }
        std::string                 getNewickString(void);
        int                         getNumNodes(void) { return numNodes; }
        int                         getNumTips(void) { return numTips; }
        Node*                       getRoot(void) { return root; }
        Node*                       findTaxonNamed(std::string tName);
        void                        initializeDownPassSequence(void);
        void                        markNodesDownFromNode(Node* p);
        void                        markUpClsAsDirtyFromNode(Node* p);
        void                        markDnClsAsDirty(void);
        void                        markUpClsAsDirty(void);
        Node*                       nodeWithOffset(size_t idx) { return nodes[idx]; }
        void                        print(void);
        void                        print(Node* subtree);
        void                        rerootOnTipZero(void);
        void                        setAllFlags(bool tf);
        void                        setDescriptor(void);
    
    private:
        Node*                       addNode(void);
        void                        clone(const Tree& t);
        void                        deleteNodes(void);
        int                         indexForTaxonName(std::string& name, std::vector<std::string>& taxonNames);
        void                        passDown(Node* p);
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
