#ifndef Node_hpp
#define Node_hpp

#include <string>
#define MAX_NAME_LENGTH 32



class Node {

    public:
                                    Node(void);
                                   ~Node(void);
        void                        addDescendant(Node* p);
        void                        clean(void);
        Node*                       getAncestor(void) { return ancestor; }
        Node*                       getFirstDescendant(void) { return firstDescendant; }
        Node*                       getNextSibling(void) { return nextSibling; }
        int                         getNumDescendants(void);
        double                      getBrlen(void) { return brlen; }
        bool                        getDirtyUpCl(void) { return dirtyUpCl; }
        bool                        getDirtyDnCl(void) { return dirtyDnCl; }
        bool                        getFlag(void) { return flag; }
        int                         getIndex(void) { return index; }
        bool                        getIsTip(void) { return isTip; }
        char*                       getName(void) { return name; }
        size_t                      getOffset(void) { return offset; }
        Node*                       getSister(void);
        void                        removeDescendant(Node* p);
        void                        removeAllDescendants(void);
        void                        setAncestor(Node* p) { ancestor = p; }
        void                        setFirstChild(Node* p) { firstDescendant = p; }
        void                        setNextSibling(Node* p) { nextSibling = p; }
        void                        setBrlen(double x) { brlen = x; }
        void                        setDirtyUpCl(bool tf) { dirtyUpCl = tf; }
        void                        setDirtyDnCl(bool tf) { dirtyDnCl = tf; }
        void                        setFlag(bool tf) { flag = tf; }
        void                        setIndex(int x) { index = x; }
        void                        setIsTip(bool tf) { isTip = tf; }
        void                        setName(char* s);
        void                        setName(std::string s);
        void                        setOffset(int x) { offset = x;}
        int                         scratchInt;
        
    private:
        Node*                       ancestor;           // ancestor node
        Node*                       firstDescendant;    // leftmost descendant (nullptr if leaf)
        Node*                       nextSibling;        // next sibling (nullptr if last descendant)
        double                      brlen;
        size_t                      offset;
        int                         index;
        char                        name[MAX_NAME_LENGTH];
        bool                        flag;
        bool                        isTip;
        bool                        dirtyUpCl;
        bool                        dirtyDnCl;
};

#endif
