#include "Alignment.hpp"
#include "ConditionalLikelihoods.hpp"



ConditionalLikelihoods::ConditionalLikelihoods(Alignment& aln) {

    taxonId = -1;
    numSites = aln.getNumSites();
    clNodeSize = numSites * 4;

    clsUp = new double[clNodeSize];
    clsDn = new double[clNodeSize];
    for (size_t i=0; i<clNodeSize; i++)
        {
        clsUp[i] = 0.0;
        clsDn[i] = 0.0;
        }
    clsEndUp = clsUp + clNodeSize;
    clsEndDn = clsDn + clNodeSize;
}

ConditionalLikelihoods::ConditionalLikelihoods(Alignment& aln, int tId) {

    taxonId = tId;
    numSites = aln.getNumSites();
    clNodeSize = numSites * 4;

    clsUp = new double[clNodeSize];
    clsDn = new double[clNodeSize];
    for (size_t i=0; i<clNodeSize; i++)
        {
        clsUp[i] = 0.0;
        clsDn[i] = 0.0;
        }
    clsEndUp = clsUp + clNodeSize;
    clsEndDn = clsDn + clNodeSize;

    double* pUp = clsUp;
    double* pDn = clsDn;
    for (size_t j=0; j<numSites; j++)
        {
        int nucCode = aln.getNucleotide(taxonId, j);
        int nucs[4];
        aln.getPossibleNucs(nucCode, nucs);
        for (size_t k=0; k<4; k++)
            {
            if (nucs[k] == 1)
                {
                *pUp = 1.0;
                *pDn = 1.0;
                }
            pUp++;
            pDn++;
            }
        }
}

ConditionalLikelihoods::~ConditionalLikelihoods(void) {

    delete [] clsUp;
    delete [] clsDn;
}
