#include <algorithm>
#include <iomanip>
#include <iostream>
#include "Alignment.hpp"
#include "ExhaustiveSearch.hpp"
#include "LikelihoodCalculator.hpp"
#include "Msg.hpp"
#include "Node.hpp"
#include "NodeFactory.hpp"
#include "Threads.hpp"
#include "Tree.hpp"



ExhaustiveSearch::ExhaustiveSearch(Alignment* aln, TreeCache* tc, ThreadPool* tp) : 
    alignment(aln), treeCache(tc), threadPool(tp), numTrees(0), scratchRoot(nullptr) {

    taxonNames = alignment->getTaxonNames();
    numTaxa = static_cast<int>(taxonNames.size());
    treesEnumerated = false;
    
    enumerateAllTrees();
}

ExhaustiveSearch::~ExhaustiveSearch(void) {

    returnScratchNodes();
}

/*-------| buildStartingTree |------------------------------------------
|   Build a 3-taxon starting tree using scratch Nodes from the
|   NodeFactory. The tree is rooted at tip 0 (the outgroup):
|
|       tip0 (root)
|         └── intNode
|               ├── tip1
|               └── tip2
|
|   For fewer than 3 taxa, degenerate trees are constructed.
*/
void ExhaustiveSearch::buildStartingTree(void) {

    returnScratchNodes();
    
    NodeFactory& nf = NodeFactory::nodeFactory();
    
    // create tip 0 (root/outgroup)
    Node* tip0 = nf.getNode();
    tip0->setIndex(0);
    tip0->setName(taxonNames[0]);
    tip0->setIsTip(true);
    scratchNodes.push_back(tip0);
    scratchRoot = tip0;
    
    if (numTaxa < 2)
        return;
    
    // create tip 1
    Node* tip1 = nf.getNode();
    tip1->setIndex(1);
    tip1->setName(taxonNames[1]);
    tip1->setIsTip(true);
    scratchNodes.push_back(tip1);
    
    if (numTaxa == 2)
        {
        // simple two-taxon tree: tip0 -> tip1
        tip0->addDescendant(tip1);
        tip1->setAncestor(tip0);
        return;
        }
    
    // create tip 2
    Node* tip2 = nf.getNode();
    tip2->setIndex(2);
    tip2->setName(taxonNames[2]);
    tip2->setIsTip(true);
    scratchNodes.push_back(tip2);
    
    // create the first internal node
    Node* intNode = nf.getNode();
    intNode->setIsTip(false);
    scratchNodes.push_back(intNode);
    
    // wire up: tip0 -> intNode -> { tip1, tip2 }
    tip0->addDescendant(intNode);
    intNode->setAncestor(tip0);
    intNode->addDescendant(tip1);
    intNode->addDescendant(tip2);
    tip1->setAncestor(intNode);
    tip2->setAncestor(intNode);
}

/*-------| collectBranches |--------------------------------------------
|   Collect all nodes except the root. Each non-root node represents
|   the branch from that node to its ancestor -- a valid insertion
|   point for a new taxon.
*/
void ExhaustiveSearch::collectBranches(Node* p, std::vector<Node*>& branches) {

    if (p == nullptr)
        return;
    if (p != scratchRoot)
        branches.push_back(p);
    for (Node* d = p->getFirstDescendant(); d != nullptr; d = d->getNextSibling())
        collectBranches(d, branches);
}

/*-------| enumerate |--------------------------------------------------
|   Recursively enumerate all unrooted binary tree topologies by
|   inserting each remaining taxon (starting at index nextTaxonIdx)
|   on every possible branch of the current scratch tree.
|
|   For a tree with k taxa already placed, there are 2k - 3 branches,
|   so the total number of trees for n taxa is:
|       (2*3-3) * (2*4-3) * ... * (2*n-3) = (2n-5)!!
|
|   At each insertion, a new tip node and a new internal node are
|   obtained from the NodeFactory, grafted onto the branch, and then
|   removed and returned to the pool after the recursive call returns.
*/
void ExhaustiveSearch::enumerate(int nextTaxonIdx, TreeCallback& callback) {

    if (nextTaxonIdx >= numTaxa)
        {
        // the scratch tree is complete -- invoke the callback
        numTrees++;
        callback(scratchRoot, numTrees);
        return;
        }
    
    // collect every branch in the current tree
    std::vector<Node*> branches;
    collectBranches(scratchRoot, branches);
    
    NodeFactory& nf = NodeFactory::nodeFactory();
    
    // get a new tip node for the taxon being added
    Node* newTip = nf.getNode();
    newTip->setIndex(nextTaxonIdx);
    newTip->setName(taxonNames[nextTaxonIdx]);
    newTip->setIsTip(true);
    scratchNodes.push_back(newTip);
    
    // get a new internal node to serve as the branching point
    Node* newInt = nf.getNode();
    newInt->setIsTip(false);
    scratchNodes.push_back(newInt);
    
    for (Node* p : branches)
        {
        Node* pAnc = p->getAncestor();
        
        // insert newInt on the branch from p to pAnc,
        // making p and newTip the two descendants of newInt
        pAnc->removeDescendant(p);
        pAnc->addDescendant(newInt);
        newInt->setAncestor(pAnc);
        newInt->addDescendant(p);
        newInt->addDescendant(newTip);
        p->setAncestor(newInt);
        newTip->setAncestor(newInt);
        
        // recurse to add the next taxon
        enumerate(nextTaxonIdx + 1, callback);
        
        // undo the insertion, restoring the scratch tree
        newTip->setAncestor(nullptr);
        p->setAncestor(pAnc);
        newInt->removeAllDescendants();
        newInt->setAncestor(nullptr);
        pAnc->removeDescendant(newInt);
        pAnc->addDescendant(p);
        }
    
    // return the temporary nodes to the pool
    scratchNodes.pop_back();  // newInt
    scratchNodes.pop_back();  // newTip
    newInt->clean();
    newTip->clean();
    nf.returnToPool(newInt);
    nf.returnToPool(newTip);
}

/*-------| search |-----------------------------------------------------
|   Run the exhaustive enumeration and invoke the callback for each
|   tree topology found. The callback receives a pointer to the root
|   of the scratch tree (which it must not modify) and the 1-based
|   tree number.
|
|   To create a Tree object from the scratch nodes, use:
|       Tree* t = new Tree(root, taxonNames);
|   which clones the node structure.
*/
void ExhaustiveSearch::search(TreeCallback callback) {

    numTrees = 0;
    buildStartingTree();
    
    if (numTaxa <= 3)
        {
        // for 1, 2, or 3 taxa there is exactly one unrooted topology
        numTrees++;
        callback(scratchRoot, numTrees);
        }
    else
        {
        // recursively add taxa 3, 4, ..., numTaxa-1
        enumerate(3, callback);
        }
    
    returnScratchNodes();
}

/*-------| searchAndCollect |-------------------------------------------
|   Convenience method that collects all Tree objects into a vector.
|   Each Tree is heap-allocated via new; the caller is responsible
|   for deleting them.
|
|   WARNING: for large numbers of taxa, this can consume enormous
|   memory. Use the callback-based search() for large problems.
*/
std::vector<Tree*> ExhaustiveSearch::searchAndCollect(void) {

    std::vector<Tree*> trees;
    search([&trees, this](Node* root, int treeNum) {
        Tree* t = new Tree(root, taxonNames);
        trees.push_back(t);
    });
    return trees;
}

void ExhaustiveSearch::enumerateAllTrees(void) {

    // enumerate all of the trees, adding each to the TreeList object
    search([this](Node* root, int treeNum) {
        Tree* t = new Tree(root, taxonNames);
        TreeInfo* tInfo = getOrCreateTreeInfo(treeCache, t);
        if (tInfo->tree != nullptr)
            Msg::error("Not expecting to find existing tree during exhaustive search");
        tInfo->tree = t;
    });
    
    // initialize the LikelihoodCalculator objects
    size_t numTrees = treeCache->size();
    size_t maxJobs = threadPool->getQueueCapacity();
    if (numTrees < maxJobs)
        maxJobs = treeCache->size();
        
    std::vector<LikelihoodCalculator> calculators;
    calculators.reserve(maxJobs);
    for (size_t i=0; i<maxJobs; i++) 
        calculators.emplace_back(alignment);
        
    // calculate the maximum likelihood for each tree in treeList
    int barWidth = 60, numAsterices = 0;
    std::cout << "   Maximum likelihood estimation for all trees:" << std::endl;
    std::cout << "   * [";
    for (int i=0; i<barWidth; i++) 
        {
        if ((i+1) % (int)(barWidth*0.1) == 0 && i+1 != barWidth)
            std::cout << "|";
        else
            std::cout << "-";
        }
    std::cout << "]" << std::endl;
    std::cout << "   * [";
    
    size_t cnt = 0, treeCnt = 0;
    bestLnL = 1.0;
    for (auto& [key,val] : *treeCache)
        {
        calculators[cnt].setTree(val->tree);
        calculators[cnt].setTreeInfo(val);
        calculators[cnt].setOffset(0);
        threadPool->pushTask(&calculators[cnt]);
        cnt++;
        treeCnt++;
        
        if (cnt == maxJobs || treeCnt == numTrees)
            {
            threadPool->wait();
            
			// print ASCII progress bar
            double progress = static_cast<double>(treeCnt) / numTrees;
            int filledWidth = static_cast<int>(progress * barWidth);
            
            //std::cout << "\r[";
            for (int i = 0; i < filledWidth-numAsterices; i++) 
                {
                std::cout << "*" << std::flush;
                }
            numAsterices = filledWidth;
                                  
            for (size_t i=0; i<cnt; i++)
                {
                double x = calculators[i].getResult();
                if (bestLnL > 0.0)
                    bestLnL = x;
                else if (x > bestLnL)
                    bestLnL = x;
                calculators[i].getTreeInfo()->lnLikelihood = x;
                calculators[i].getTreeInfo()->hasLnLikelihood = true;
                }
            cnt = 0;
            }
        }
        
    // print final newline after progress bar completion
    std::cout << "]" << std::endl << std::endl;
    
    treesEnumerated = true;
}

/*-------| returnScratchNodes |-----------------------------------------
|   Return all scratch nodes to the NodeFactory pool and clear
|   the tracking vector.
*/
void ExhaustiveSearch::returnScratchNodes(void) {

    NodeFactory& nf = NodeFactory::nodeFactory();
    for (Node* n : scratchNodes)
        {
        n->clean();
        nf.returnToPool(n);
        }
    scratchNodes.clear();
    scratchRoot = nullptr;
}

/*-------| numUnrootedTrees |-------------------------------------------
|   Return the number of unrooted binary trees for n taxa:
|       (2n - 5)!! = 1 * 3 * 5 * ... * (2n - 5)
|   Returns 1 for n <= 3.
*/
int ExhaustiveSearch::numUnrootedTrees(int n) {

    if (n <= 3)
        return 1;
    int count = 1;
    for (int i = 3; i <= n; i++)
        count *= (2 * i - 5);
    return count;
}
