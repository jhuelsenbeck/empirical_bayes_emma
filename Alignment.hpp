#ifndef Alignment_H
#define Alignment_H

#include <bitset>
#include <string>
#include <vector>
#include "ncl.h"



class Alignment {

    public:
                                                    Alignment(void) = delete;
                                                    Alignment(Alignment& a) = delete;
                                                    Alignment(std::string fileName);
                                                    Alignment(std::vector<std::string> tn, int nr, int nc);
                                                   ~Alignment(void);
        int&                                        operator()(size_t r, size_t c) { return this->matrix[r][c]; }
        const int&                                  operator()(size_t r, size_t c) const { return this->matrix[r][c]; }
        int                                         getNumTaxa(void) { return numTaxa; }
        int                                         getNumSites(void);
        int                                         getNumStates(void);
        void                                        getPossibleNucs (int nucCode, int* nuc);
        int                                         getNucleotide(size_t i, size_t j);
        int                                         getTaxonIndex(std::string ns);
        std::vector<std::string>                    getTaxonNames(void);
        std::string                                 getTaxonName(int i);
        int                                         lengthOfLongestTaxonName(void);
        void                                        listTaxa(void);
        void                                        print(void);
        void                                        summarize(void);

    private:
        void                                        createDnaMatrix(NxsCharactersBlock* charblock);
        int                                         nucID(char nuc);
        int**                                       matrix;
        int                                         numTaxa;
        int                                         numNucleotideSites;
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
