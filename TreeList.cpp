#include <iomanip>
#include <iostream>
#include "Msg.hpp"
#include "Tree.hpp"
#include "TreeList.hpp"



TreeList::~TreeList(void) {

    for (TreeMapIter it=map.begin(); it != map.end(); it++)
        delete it->second.tree;
}

double TreeList::lnLikelihood(Tree* t) {

    TreeMap::iterator it = map.find(t->getHash());
    if (it == map.end())
        {
        TreeInfo info(t,0.0,false);
        map.insert(std::make_pair(t->getHash(),info));
        }
    
    return it->second.lnL;
}

void TreeList::addTree(Tree* t) {

    TreeInfo info(t,0.0,false);
    map.insert(std::make_pair(t->getHash(),info));
}

void TreeList::addTree(Tree* t, double x) {

    TreeInfo info(t,x,true);
    map.insert(std::make_pair(t->getHash(),info));
}

Tree* TreeList::getTree(uint64_t treeHash) {

    TreeMapIter it = map.find(treeHash);
    if (it == map.end())
        return nullptr;
    return it->second.tree;
}

TreeInfo& TreeList::getTreeInfo(uint64_t treeHash) {

    TreeMapIter it = map.find(treeHash);
    if (it == map.end())
        Msg::error("Could not find tree in tree list");
    return it->second;
}

void TreeList::print(void) {

    int i = 1;
    for (TreeMapIter it = map.begin(); it != map.end(); it++)
        std::cout << std::setw(6) << i++ << " " << std::setw(20) << it->first << " " << it->second.lnL << std::endl;
}
