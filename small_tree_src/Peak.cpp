#include "Peak.hpp"



Peak::Peak(void) : 
    peakTreeHash(0), peakTreeProbability(0.0), peakProbability(0.0), numTrees(0), peakId(0) {

}

bool Peak::isTreeInPeak(uint64_t treeHash) {

    TreeSet::iterator it = memberTrees.find(treeHash);
    if (it != memberTrees.end())
        return true;
    return false;
}
