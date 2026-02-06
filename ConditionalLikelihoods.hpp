#ifndef ConditionalLikelihoods_hpp
#define ConditionalLikelihoods_hpp

#include <cstddef>
class Alignment;



class ConditionalLikelihoods {

    public:
                    ConditionalLikelihoods(void) = delete;
                    ConditionalLikelihoods(Alignment& aln);
                    ConditionalLikelihoods(Alignment& aln, int tId);
                   ~ConditionalLikelihoods(void);
        double*     beginUp(void) { return clsUp; }
        double*     beginDn(void) { return clsDn; }
        double*     endUp(void) { return clsEndUp; }
        double*     endDn(void) { return clsEndDn; }
    
    private:
        double*     clsUp;
        double*     clsDn;
        double*     clsEndUp;
        double*     clsEndDn;
        int         taxonId;
        size_t      numSites;
        size_t      clNodeSize;
};

#endif
