#include "TreeSamples.hpp"
#include <vector>
#include <algorithm>
#include <iostream>



TreeSamples::TreeSamples(void) : numSamples(0) {

}

void TreeSamples::print(void) {

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

    // print sorted results: "hash count"
    double sum = 0.0;
    for (const auto &e : entries)
        {
        double prob = (double)e.second / numSamples;
        sum += prob;
        std::cout << e.first << " " << prob << " " << sum << '\n';
        }
}

void TreeSamples::sampleTree(uint64_t treeHash) {

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
