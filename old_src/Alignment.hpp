#ifndef Alignment_H
#define Alignment_H

#include <bitset>
#include <string>
#include <vector>
#include <map>
#include "ncl.h"
class RandomVariable;



class Alignment {

    public:
                                                    Alignment(void) = delete;
                                                    Alignment(Alignment& a) = delete;
                                                    Alignment(std::string fileName);
                                                    Alignment(std::vector<std::string> tn, int nr, int nc);
                                                    Alignment(Alignment& a, int nt, RandomVariable* rng);
                                                    Alignment(std::vector<std::string> tn, std::string newickString, int nc, RandomVariable* rng);
                                                   ~Alignment(void);
        int&                                        operator()(size_t r, size_t c) { return this->matrix[r][c]; }
        const int&                                  operator()(size_t r, size_t c) const { return this->matrix[r][c]; }
        void                                        compress(void);
        int                                         getNumTaxa(void) { return numTaxa; }
        int                                         getNumSites(void) { return numSites; }
        int                                         getNumStates(void) { return 4; }
        int                                         getPatternCount(int i) { return (patternCount != nullptr && i >= 0 && i < numSites) ? patternCount[i] : 1; }
        void                                        getPossibleNucs (int nucCode, int* nuc);
        int                                         getNucleotide(size_t i, size_t j) { return matrix[i][j]; }
        int                                         getTaxonIndex(std::string ns);
        std::vector<std::string>&                   getTaxonNames(void) { return taxonNames; }
        std::string&                                getTaxonName(int i) { return taxonNames[i]; }
        int                                         lengthOfLongestTaxonName(void);
        void                                        listTaxa(void);
        void                                        print(void);
        void                                        print(std::string fileName);
        void                                        summarize(void);
        void                                        twist(RandomVariable* rng, int nTrees);

    private:
        void                                        createDnaMatrix(NxsCharactersBlock* charblock);
        int                                         nucID(char nuc);
        char                                        toNuc(int charCode);
        int**                                       matrix;
        int*                                        patternCount;
        int                                         numTaxa;
        int                                         numSites;
        bool                                        isCompressed;
        std::vector<std::string>                    taxonNames;
};

inline std::ostream& operator<<(std::ostream& os, Alignment& a) {

    int len = a.lengthOfLongestTaxonName();
    for (int i=0; i<a.getNumTaxa(); i++)
        {
        os << a.getTaxonName(i);
        os << " ";
        for (int j=0; j<len-a.getTaxonName(i).length(); j++)
            os << " ";
        for (int j=0; j<a.getNumSites(); j++)
            {
            int x = a(i,j);
            if (x == 0)
                os << 'A';
            else if (x == 1)
                os << 'C';
            else if (x == 2)
                os << 'G';
            else
                os << 'T';
            }
        os << std::endl;
        }

    return os;
}

#endif
