#include <algorithm>
#include <iomanip>
#include <iostream>
#include <set>
#include <vector>
#include "BitSet.hpp"
#include "BitSetFactory.hpp"
#include "Node.hpp"
#include "Tree.hpp"
#include "TreePartitions.hpp"



TreePartitions::TreePartitions(int n) : numTaxa(n), count(0) {

}

TreePartitions::~TreePartitions(void) {

    for (PartitionMap::iterator it = taxonBipartitions.begin(); it != taxonBipartitions.end(); it++)
        delete it->first;
}

void TreePartitions::addTree(Tree* t) {

    count++;

    int numNodes = t->getNumNodes();
    int numTips = t->getNumTips();

    // get bit sets for each node
    BitSetFactory& bf = BitSetFactory::getFactory();
    std::vector<BitSet*> partitions(numNodes);
    for (size_t i=0; i<numNodes; i++)
        partitions[i] = bf.getBitSet();

    // initialize the bit set for the tip nodes
    for (size_t i=0; i<numTips; i++)
        partitions[i]->set(i);
        
    std::set<BitSet*,CompBitSet> nonTrivialPartitions;
    const std::vector<Node*>& downPassSequence = t->getDownPassSequence();
    Node* root = t->getRoot();
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
                
    for (std::set<BitSet*,CompBitSet>::iterator it = nonTrivialPartitions.begin(); it != nonTrivialPartitions.end(); it++)
        {
        PartitionMap::iterator p = taxonBipartitions.find(*it);
        if (p == taxonBipartitions.end())
            taxonBipartitions.insert( std::make_pair(new BitSet(**it), 1) );
        else    
            p->second++;
        }
    
    for (size_t i=0; i<partitions.size(); i++)
        bf.returnToPool(partitions[i]);
}

void TreePartitions::comparePartitions(std::vector<TreePartitions*>& parts) {

    size_t numChains = parts.size();
    if (numChains < 2)
        {
        std::cout << "Need at least two chains to compare split frequencies" << std::endl;
        return;
        }

    // all chains must agree on taxon count, otherwise pointwise frequency
    // comparison across chains is meaningless
    int n = parts[0]->numTaxa;
    for (size_t i=1; i<numChains; i++)
        {
        if (parts[i]->numTaxa != n)
            {
            std::cout << "Cannot compare: chains have different taxon counts" << std::endl;
            return;
            }
        }

    // CompBitSet orders by content, so the same split appearing in multiple
    // chains collapses to a single set entry regardless of which chain's
    // BitSet* we happened to pick up
    std::set<BitSet*,CompBitSet> uniquePartitions;
    for (size_t i=0; i<numChains; i++)
        {
        for (auto& [key,val] : parts[i]->taxonBipartitions)
            uniquePartitions.insert(key);
        }

    struct SplitStats
        {
        BitSet*             split;
        std::vector<double> freqs;
        double              mean;
        double              stdDev;
        double              cv;
        double              minF;
        double              maxF;
        };

    std::vector<SplitStats> rows;
    rows.reserve(uniquePartitions.size());

    for (BitSet* p : uniquePartitions)
        {
        SplitStats r;
        r.split = p;
        r.freqs.assign(numChains, 0.0);

        // missing from a chain == observed zero times in that chain
        for (size_t i=0; i<numChains; i++)
            {
            PartitionMap::iterator it = parts[i]->taxonBipartitions.find(p);
            if (it != parts[i]->taxonBipartitions.end())
                r.freqs[i] = (double)it->second / parts[i]->count;
            }

        double sum = 0.0;
        for (double f : r.freqs)
            sum += f;
        r.mean = sum / numChains;

        // Bessel-corrected sample variance; for two chains this is just
        // (f1 - f2)^2 / 2 -- noisy but matches MrBayes' ASDSF convention
        double sse = 0.0;
        for (double f : r.freqs)
            {
            double d = f - r.mean;
            sse += d * d;
            }
        r.stdDev = std::sqrt(sse / (numChains - 1));

        // CV undefined at zero mean; the split is in the set so at least one
        // chain saw it, but report 0 defensively
        r.cv = (r.mean > 0.0) ? (r.stdDev / r.mean) : 0.0;

        r.minF = r.freqs[0];
        r.maxF = r.freqs[0];
        for (size_t i=1; i<numChains; i++)
            {
            if (r.freqs[i] < r.minF) r.minF = r.freqs[i];
            if (r.freqs[i] > r.maxF) r.maxF = r.freqs[i];
            }

        rows.push_back(std::move(r));
        }

    // most-probable splits at the top
    std::sort(rows.begin(), rows.end(), [](const SplitStats& a, const SplitStats& b) {
        return a.mean > b.mean;
    });

    // ASDSF/MSDSF restricted to splits credible in at least one chain;
    // without this restriction the average is swamped by zero-variance
    // never-seen splits and stops being informative
    const double threshold = 0.1;
    double sumStdDev = 0.0;
    double maxStdDev = 0.0;
    size_t numAbove  = 0;
    for (const SplitStats& r : rows)
        {
        if (r.maxF >= threshold)
            {
            sumStdDev += r.stdDev;
            if (r.stdDev > maxStdDev)
                maxStdDev = r.stdDev;
            numAbove++;
            }
        }
    double asdsf = (numAbove > 0) ? sumStdDev / numAbove : 0.0;

    // per-split table
    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::setw(5) << "#" << "  ";
    std::cout << std::setw(n) << std::left << "split" << std::right;
    for (size_t i=0; i<numChains; i++)
        {
        std::string label = "chain" + std::to_string(i+1);
        std::cout << "  " << std::setw(8) << label;
        }
    std::cout << "  " << std::setw(8) << "mean";
    std::cout << "  " << std::setw(8) << "stdev";
    std::cout << "  " << std::setw(8) << "cv";
    std::cout << "  " << std::setw(8) << "min";
    std::cout << "  " << std::setw(8) << "max";
    std::cout << '\n';

    int idx = 0;
    for (const SplitStats& r : rows)
        {
        std::cout << std::setw(5) << ++idx << "  ";
        std::cout << std::setw(n) << std::left << r.split->bitString() << std::right;
        for (double f : r.freqs)
            std::cout << "  " << std::setw(8) << f;
        std::cout << "  " << std::setw(8) << r.mean;
        std::cout << "  " << std::setw(8) << r.stdDev;
        std::cout << "  " << std::setw(8) << r.cv;
        std::cout << "  " << std::setw(8) << r.minF;
        std::cout << "  " << std::setw(8) << r.maxF;
        std::cout << '\n';
        }

    // cross-chain summary
    std::cout << '\n';
    std::cout << "Convergence diagnostics for split frequencies\n";
    std::cout << "   Number of chains          = " << numChains << '\n';
    std::cout << "   Total unique splits       = " << rows.size() << '\n';
    std::cout << "   Splits with max p >= " << threshold << " = " << numAbove << '\n';
    std::cout << "   ASDSF (max p >= " << threshold << ")    = " << asdsf << '\n';
    std::cout << "   MSDSF (max p >= " << threshold << ")    = " << maxStdDev << '\n';
}

void TreePartitions::print(void) {

    // collect entries from the map
    std::vector<std::pair<BitSet*, uint32_t>> entries;
    entries.reserve(taxonBipartitions.size());
    for (const auto &kv : taxonBipartitions) 
        entries.emplace_back(kv.first, kv.second);

    // sort by count descending; tie-break by hash ascending for determinism
    std::sort(entries.begin(), entries.end(), [](const auto &a, const auto &b) {
        if (a.second != b.second) return a.second > b.second; // higher counts first
        return a.first < b.first; // smaller hash first on ties
    });

    // print sorted results
    int i = 0;
    for (const auto &e : entries)
        {
        double prob = (double)e.second / count;
        std::cout << std::setw(5) << ++i << " -- ";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << std::setw(21) << e.first->bitString() << " " << prob << '\n';
        if (prob < 0.1)
            break;
        }
}

void TreePartitions::writeStatsHeader(std::ostream& os) {

    os << "numUniqueSplits\tnumSplitsAboveThr\tASDSF\tMSDSF";
}

void TreePartitions::writeStatsLine(std::ostream& os, std::vector<TreePartitions*>& parts) {

    size_t numChains = parts.size();
    if (numChains < 2)
        {
        // emit zeros so column count stays fixed across the run
        os << "0\t0\t0\t0";
        return;
        }

    // gather unique splits across chains; same content collapses regardless
    // of which chain's BitSet* we pick up (CompBitSet orders by content)
    std::set<BitSet*,CompBitSet> uniquePartitions;
    for (size_t i=0; i<numChains; i++)
        for (auto& [key,val] : parts[i]->taxonBipartitions)
            uniquePartitions.insert(key);

    const double threshold = 0.1;
    double sumStdDev = 0.0;
    double maxStdDev = 0.0;
    size_t numAbove  = 0;

    std::vector<double> freqs(numChains);
    for (BitSet* p : uniquePartitions)
        {
        for (size_t i=0; i<numChains; i++)
            {
            PartitionMap::iterator it = parts[i]->taxonBipartitions.find(p);
            freqs[i] = (it != parts[i]->taxonBipartitions.end())
                       ? (double)it->second / parts[i]->count : 0.0;
            }

        double mean = 0.0;
        double maxF = freqs[0];
        for (size_t i=0; i<numChains; i++)
            {
            mean += freqs[i];
            if (freqs[i] > maxF) maxF = freqs[i];
            }
        mean /= numChains;

        // gating at threshold matches the ASDSF/MSDSF computation in
        // comparePartitions; keeps the time series interpretable
        if (maxF < threshold)
            continue;

        double sse = 0.0;
        for (double f : freqs)
            {
            double d = f - mean;
            sse += d * d;
            }
        double sd = std::sqrt(sse / (numChains - 1));
        sumStdDev += sd;
        if (sd > maxStdDev)
            maxStdDev = sd;
        numAbove++;
        }

    double asdsf = (numAbove > 0) ? sumStdDev / numAbove : 0.0;

    os << uniquePartitions.size() << '\t' << numAbove
       << '\t' << asdsf << '\t' << maxStdDev;
}
