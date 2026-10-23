#include <algorithm>
#include <iostream>
#include <set>
#include "CompactTree.hpp"
#include "Msg.hpp"
#include "Node.hpp"
#include "Threads.hpp"
#include "Tree.hpp"
#include "TreeNeighborGenerator.hpp"
#define XXH_INLINE_ALL
#include "xxhash.h"


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
        TreeInfo* ti1 = treeCache->getOrCreateTreeInfo(t1);
        TreeInfo* ti2 = treeCache->getOrCreateTreeInfo(t2);
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

TreeNeighborGeneratorNNI2::TreeNeighborGeneratorNNI2(TreeCache* tc, TreeCache* nniTc) : TreeNeighborGenerator(tc), nniTreeCache(nniTc) {

}

void TreeNeighborGeneratorNNI2::generateNeighbors(Tree* tree, std::vector<TreeInfo*>& neighbors) {

    generateNeighbors(tree->getHash(), neighbors);
}

void TreeNeighborGeneratorNNI2::generateNeighbors(uint64_t treeHash, std::vector<TreeInfo*>& neighbors) {

    // get the degree one neighbors
    TreeInfo* info = nniTreeCache->getTreeInfo(treeHash);
    if (info == nullptr)
        Msg::error("Tree info could not be found in TreeNeighborGeneratorNNI2");
    NeighborList degreeOneNeighbors = nniTreeCache->neighbors(info);
    if (degreeOneNeighbors.size() == 0)
        Msg::error("No neighbors found in TreeNeighborGeneratorNNI2");

    // find all degree one and degree two neighbors
    std::set<TreeInfo*> uniqueNeighbors;
    for (TreeInfo* nbr1 : degreeOneNeighbors)
        {
        uniqueNeighbors.insert(nbr1);

        NeighborList degreeTwoNeighbors = nniTreeCache->neighbors(nbr1);
        for (TreeInfo* nbr2 : degreeTwoNeighbors)
            {
            if (nbr2 != info)
                uniqueNeighbors.insert(nbr2);
            }
        }

    // add them to the neighbors vector
    neighbors.reserve(uniqueNeighbors.size());
    for (TreeInfo* nbr : uniqueNeighbors)
        {
        TreeInfo* ti = treeCache->getTreeInfo(nbr->hash);
        if (ti == nullptr)
            Msg::error("Could not find tree in cache for NNI2");
        neighbors.push_back(ti);
        }
}

TreeNeighborGeneratorTBR::TreeNeighborGeneratorTBR(TreeCache* tc) : TreeNeighborGenerator(tc) {

}

void TreeNeighborGeneratorTBR::generateNeighbors(Tree* tree, std::vector<TreeInfo*>& neighbors) {

    // generate TBR neighbors
    int nNeighbors = tree->numTbrNeighbors();
    neighbors.reserve(nNeighbors);

    const std::vector<Node*>& postOrder = tree->getDownPassSequence();
    Node* rootNode = tree->getRoot();
    uint64_t treeHash = tree->getHash();
    size_t numSelfSkipped = 0;
    
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

                // The reconnection that rejoins the two subtrees at the branch
                // ends that were just bisected reproduces the starting
                // topology. Skip it: a tree is not its own TBR neighbor.
                // Without this guard the identity reconnection is stored as a
                // self-loop, inflating every vertex degree by one (for example
                // giving degree 3 instead of 2 for the three trees of the N=4
                // graph). The skipped identities are still counted so that the
                // enumeration-completeness check below remains valid.
                if (t->getHash() == treeHash)
                    {
                    numSelfSkipped++;
                    delete t;
                    continue;
                    }

                TreeInfo* ti = treeCache->getOrCreateTreeInfo(t);
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
        
    // check results. Every enumerated (bisection, reconnection) pair is either
    // stored as a neighbor or skipped as an identity reconnection, so the two
    // counts together must equal the analytic count from numTbrNeighbors().
    if ((size_t)nNeighbors != neighbors.size() + numSelfSkipped)
        Msg::error("Did not generate the expected number of TBR neighbors");

}

TreeNeighborGeneratorRandomTBR::TreeNeighborGeneratorRandomTBR(TreeCache* tc) : TreeNeighborGenerator(tc) {

}

void TreeNeighborGeneratorRandomTBR::generateNeighbors(Tree* tree, std::vector<TreeInfo*>& neighbors) {

    Msg::error("Neighbor generation under random TBR requires a random number generator");
}

namespace {

    // A binary tree rooted on tip 0, held as cluster masks: for every node other than tip 0, the set
    // of taxa below it (bit i = taxon i), its parent (-1 for the root's child, whose parent is tip 0)
    // and its two children (-1 for a tip). There are 2n - 3 such nodes, one per edge (the edge above
    // the node), and the masks of distinct nodes are distinct and laminar.
    struct MaskTree {

        int         numNodes = 0;
        int         rootChild = -1;
        uint32_t    mask[32];
        int         parent[32];
        int         child0[32];
        int         child1[32];
    };

    // Read one subtree of the compact encoding: a tip is Tip index; an interior node is
    // Open child Close child Close (CompactTree::writeSubtree, for a binary tree).
    int parseCompact(const CompactTree* ct, size_t& pos, int taxonBits, MaskTree& t, int parentNode) {

        CompactTree::Code code = ct->readCode(pos);
        int me = t.numNodes++;
        if (me >= 32)
            Msg::error("Tree too large for the split-mask TBR generator");
        t.parent[me] = parentNode;
        if (code == CompactTree::Code::Tip)
            {
            t.mask[me] = uint32_t(1) << ct->readIndex(pos, taxonBits);
            t.child0[me] = t.child1[me] = -1;
            return me;
            }
        if (code != CompactTree::Code::Open)
            Msg::error("Unexpected code in a compact tree");
        int a = parseCompact(ct, pos, taxonBits, t, me);
        if (ct->readCode(pos) != CompactTree::Code::Close)
            Msg::error("Expecting a binary tree in the split-mask TBR generator");
        int b = parseCompact(ct, pos, taxonBits, t, me);
        if (ct->readCode(pos) != CompactTree::Code::Close)
            Msg::error("Expecting a binary tree in the split-mask TBR generator");
        t.child0[me] = a;
        t.child1[me] = b;
        t.mask[me] = t.mask[a] | t.mask[b];
        return me;
    }

    // One edge of a side's subtree after the cut: K is the side of the edge away from the cut point,
    // within that side's taxa. The edge at the cut point itself (the two edges that met there, merged
    // once the cut leaves their junction with degree two) is marked merged.
    struct SideEdge {

        uint32_t    K;
        bool        merged;
    };

    inline bool strictSubset(uint32_t a, uint32_t b) { return (a & ~b) == 0 && a != b; }

    inline int popCount(uint32_t x) { return __builtin_popcount(x); }

    class TbrSplitTask : public ThreadTask {

        public:
                                TbrSplitTask(const TbrSplitNeighbors* g, TreeCache* tc, uint32_t lo, uint32_t hi) :
                                    generator(g), treeCache(tc), first(lo), last(hi) { }
            void                run(void) override
                                    {
                                    entries.clear();
                                    counts.clear();
                                    std::vector<uint32_t> row;
                                    for (uint32_t i = first; i < last; i++)
                                        {
                                        row.clear();
                                        generator->generate(treeCache->treeAt(i), row);
                                        counts.push_back(static_cast<uint32_t>(row.size()));
                                        entries.insert(entries.end(), row.begin(), row.end());
                                        }
                                    }
            const TbrSplitNeighbors* generator;
            TreeCache*          treeCache;
            uint32_t            first;
            uint32_t            last;
            std::vector<uint32_t> entries;
            std::vector<uint32_t> counts;
    };
}

TbrSplitNeighbors::TbrSplitNeighbors(TreeCache* tc, int nt) : treeCache(tc), numTaxa(nt) {

    if (numTaxa < 4 || numTaxa > 16)
        Msg::error("The split-mask TBR generator handles 4 to 16 taxa");
}

// The hash Tree::setDescriptor forms: the clusters sorted as std::set<BitSet*, CompBitSet> orders them
// (at the lowest taxon index where two differ, the one containing that taxon first), cluster j written
// to bits j*n .. j*n + n - 1 of a BitSet (bit i for taxon i), the BitSet's words filled most significant
// bit first, and XXH3 taken over those words.
uint64_t TbrSplitNeighbors::hashOfClusters(uint32_t* clusters, int numClusters, int numTaxa) {

    std::sort(clusters, clusters + numClusters, [](uint32_t a, uint32_t b)
        {
        uint32_t d = a ^ b;
        return (a & d & (~d + 1)) != 0;                 // a holds the lowest differing taxon
        });

    unsigned words[16] = { 0 };                          // up to 13 clusters x 16 taxa = 208 bits
    size_t numBits = static_cast<size_t>(numClusters) * numTaxa;
    size_t numWords = (numBits + 31) / 32;
    for (int j = 0; j < numClusters; j++)
        {
        uint32_t c = clusters[j];
        while (c != 0)
            {
            int i = __builtin_ctz(c);
            c &= c - 1;
            size_t bit = static_cast<size_t>(j) * numTaxa + i;
            words[bit / 32] |= (0x80000000u >> (bit % 32));
            }
        }
    return XXH3_64bits(words, numWords * sizeof(unsigned));
}

void TbrSplitNeighbors::generate(const TreeInfo* info, std::vector<uint32_t>& out) const {

    MaskTree t;
    size_t pos = 0;
    int taxonBits = CompactTree::bitsPerIndex(numTaxa);
    t.rootChild = parseCompact(info->compactTree, pos, taxonBits, t, -1);
    if (info->compactTree->readCode(pos) != CompactTree::Code::Tip || info->compactTree->readIndex(pos, taxonBits) != 0)
        Msg::error("Expecting tip 0 as the root of a compact tree");

    const int n = numTaxa;
    SideEdge xEdges[32], yEdges[32];
    uint32_t splits[64], clusters[32];
    size_t expected = 0, numSelf = 0, before = out.size();

    for (int p = 0; p < t.numNodes; p++)
        {
        // cut the edge above p: Y below it, X (with tip 0) above
        const uint32_t Y = t.mask[p];

        // the Y side: every edge strictly inside Y but p's two child edges, which merge into one
        int nY = 0;
        uint32_t C1 = 0, C2 = 0;
        if (t.child0[p] >= 0)
            {
            C1 = t.mask[t.child0[p]];
            C2 = t.mask[t.child1[p]];
            for (int q = 0; q < t.numNodes; q++)
                {
                if (q == p || q == t.child0[p] || q == t.child1[p])
                    continue;
                if (strictSubset(t.mask[q], Y) == true)
                    yEdges[nY++] = { t.mask[q], false };
                }
            yEdges[nY++] = { Y, true };                // the merged edge, K marked by Y itself
            }

        // the X side: every edge outside Y but p's sibling edge and parent edge, which merge into one.
        // An edge above p keeps only its part outside Y.
        int nX = 0;
        if (p != t.rootChild)
            {
            int u = t.parent[p];
            int S = (t.child0[u] == p) ? t.child1[u] : t.child0[u];
            for (int q = 0; q < t.numNodes; q++)
                {
                if (q == p || q == S || q == u)
                    continue;
                uint32_t m = t.mask[q];
                if ((m & ~Y) == 0)
                    continue;                              // inside Y
                xEdges[nX++] = { (m & Y) != 0 ? (m & ~Y) : m, false };
                }
            xEdges[nX++] = { t.mask[S], true };
            }

        int numA = (nX > 0) ? nX : 1;                      // X = {0} is a single attachment point
        int numB = (nY > 0) ? nY : 1;                      // so is a single tip below the cut
        expected += static_cast<size_t>(numA) * numB;

        for (int ia = 0; ia < numA; ia++)
            {
            for (int ib = 0; ib < numB; ib++)
                {
                int ns = 0;
                splits[ns++] = Y;                          // the joining edge

                if (nX > 0)
                    {
                    const uint32_t Ka = xEdges[ia].K;
                    for (int f = 0; f < nX; f++)
                        {
                        uint32_t Kf = xEdges[f].K;
                        if (f == ia)
                            {
                            splits[ns++] = Ka;             // the subdivided attachment edge
                            splits[ns++] = Ka | Y;
                            }
                        else
                            splits[ns++] = (strictSubset(Ka, Kf) == true) ? (Kf | Y) : Kf;
                        }
                    }

                if (nY > 0)
                    {
                    const SideEdge& b = yEdges[ib];
                    for (int g = 0; g < nY; g++)
                        {
                        const SideEdge& e = yEdges[g];
                        if (g == ib)
                            {
                            if (b.merged == true)
                                {
                                splits[ns++] = C1;
                                splits[ns++] = C2;
                                }
                            else
                                {
                                splits[ns++] = b.K;
                                splits[ns++] = Y & ~b.K;
                                }
                            }
                        else if (e.merged == true)
                            splits[ns++] = ((b.K & ~C1) == 0) ? C2 : C1;
                        else
                            splits[ns++] = (b.merged == false && strictSubset(b.K, e.K) == true) ? (Y & ~e.K) : e.K;
                        }
                    }

                // the n - 3 nontrivial splits, by their side without tip 0 (every mask here excludes it)
                int nc = 0;
                for (int k = 0; k < ns; k++)
                    {
                    int pc = popCount(splits[k]);
                    if (pc >= 2 && pc <= n - 2)
                        clusters[nc++] = splits[k];
                    }
                if (nc != n - 3)
                    Msg::error("Split-mask TBR generator produced a tree with the wrong number of splits");

                uint64_t h = hashOfClusters(clusters, nc, n);
                if (h == info->hash)
                    {
                    numSelf++;                             // the identity reconnection
                    continue;
                    }
                TreeInfo* nbr = treeCache->getTreeInfo(h);
                if (nbr == nullptr)
                    Msg::error("Split-mask TBR generator produced a tree that is not in the cache");
                out.push_back(nbr->index);
                }
            }
        }

    if (out.size() - before + numSelf != expected)
        Msg::error("Split-mask TBR generator did not produce the expected number of neighbors");
}

void TbrSplitNeighbors::generateAll(ThreadPool* pool) {

    // Trees are handed out in blocks; each batch of blocks is generated in parallel, then appended to
    // the cache's neighbor array in tree order, so only one batch of rows is held twice at a time.
    uint32_t numTrees = static_cast<uint32_t>(treeCache->size());
    const uint32_t blockSize = 1024;
    int numThreads = (pool != nullptr) ? pool->getThreadCount() : 1;
    uint32_t blocksPerBatch = static_cast<uint32_t>(std::max(1, numThreads * 4));

    std::cout << "   Generating TBR neighbors for all " << numTrees << " trees from split masks (" << numThreads << " threads)" << std::endl;
    std::vector<TbrSplitTask> tasks;
    for (uint32_t start = 0; start < numTrees; start += blockSize * blocksPerBatch)
        {
        tasks.clear();
        for (uint32_t lo = start; lo < numTrees && lo < start + blockSize * blocksPerBatch; lo += blockSize)
            tasks.emplace_back(this, treeCache, lo, std::min(numTrees, lo + blockSize));
        if (pool != nullptr)
            {
            for (TbrSplitTask& task : tasks)
                pool->pushTask(&task);
            pool->wait();
            }
        else
            {
            for (TbrSplitTask& task : tasks)
                task.run();
            }
        for (TbrSplitTask& task : tasks)
            {
            const uint32_t* row = task.entries.data();
            for (uint32_t k = 0; k < task.counts.size(); k++)
                {
                treeCache->setNeighbors(treeCache->treeAt(task.first + k), row, task.counts[k]);
                row += task.counts[k];
                }
            }
        }
    treeCache->finishNeighbors();
}
