#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>
#include "TreeList.hpp"
#include "TreeSamples.hpp"



TreeSamples::TreeSamples(TreeList* tl) : treeList(tl), numSamples(0) {

}

double TreeSamples::getTreeProbability(uint64_t treeHash) {

    TreeCountMap::iterator it = treeCounts.find(treeHash);
    if (it != treeCounts.end())
        return (double)it->second / numSamples;
    return 0.0;
}

void TreeSamples::print(std::map<uint64_t,std::pair<double,double>>& treeProbabilities) {

    // collect entries from the map
    std::vector<std::pair<uint64_t, std::size_t>> entries;
    entries.reserve(treeCounts.size());
    for (const auto &kv : treeCounts) 
        entries.emplace_back(kv.first, kv.second);

    // sort by count descending; tie-break by hash ascending for determinism
    std::sort(entries.begin(), entries.end(), [](const auto &a, const auto &b) {
        if (a.second != b.second) return a.second > b.second; // higher counts first
        return a.first < b.first; // smaller hash first on ties
    });

    // print sorted results
    double sum = 0.0;
    int i = 0;
    for (const auto &e : entries)
        {
        double lnL = treeList->getTreeInfo(e.first).lnL;
        double prob = (double)e.second / numSamples;
        sum += prob;
        std::cout << std::setw(5) << ++i << " -- ";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << std::setw(21) << e.first << " " << lnL << " " << prob << " " << sum << '\n';
        
        std::map<uint64_t,std::pair<double,double>>::iterator it = treeProbabilities.find(e.first);
        if (it == treeProbabilities.end())
            treeProbabilities.insert( std::make_pair(e.first,std::make_pair(0.0,prob)) );
        else 
            it->second.second = prob;
            
        if (sum > 0.99)
            break;
        }
}

void TreeSamples::sampleTree(uint64_t treeHash) {

    numSamples++;
    TreeCountMap::iterator it = treeCounts.find(treeHash);
    if (it == treeCounts.end())
        {
        treeCounts.insert(std::make_pair(treeHash,1));
        }
    else 
        {
        it->second++;
        }
}
