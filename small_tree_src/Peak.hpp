#ifndef Peak_hpp
#define Peak_hpp

#include <cmath>
#include <cstdint>
#include <unordered_set>
typedef std::unordered_set<uint64_t> TreeSet;


class Peak {

    public:
                    Peak(void);
        void        addTreeToPeak(uint64_t treeHash, double x) { peakProbability += x; numTrees++; memberTrees.insert(treeHash); }
        int         getNumTrees(void) { return numTrees; }
        int         getPeakId(void) { return peakId; }
        double      getPeakProbability(void) { return peakProbability; }
        double      getPeakTreeProbability(void) { return peakTreeProbability; }
        uint64_t    getPeakTreeHash(void) { return peakTreeHash; }
        TreeSet&    getTrees(void) { return memberTrees; }
        void        setPeakId(int x) { peakId = x; }
        bool        isTreeInPeak(uint64_t treeHash);
        void        setPeakTreeHash(uint64_t x) { peakTreeHash = x; }
        void        setPeakTreeProbability(double x) { peakTreeProbability = x; }
    
    private:
        uint64_t    peakTreeHash;
        double      peakTreeProbability;
        double      peakProbability;
        int         numTrees;
        int         peakId;
        TreeSet     memberTrees;
};

#endif
