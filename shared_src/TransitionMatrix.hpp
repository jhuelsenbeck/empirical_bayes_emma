#ifndef TransitionMatrix_hpp
#define TransitionMatrix_hpp



class TransitionMatrix {

    public:
                        TransitionMatrix(void);
        virtual        ~TransitionMatrix(void) = default;
        double&         operator()(size_t r, size_t c) { return this->vals[4 * r + c]; }
        const double&   operator()(size_t r, size_t c) const { return this->vals[4 * r + c]; }
        double*         begin(void) { return vals; }
        double*         end(void) { return valsEnd; }
        double          getBrlen(void) { return brlen; }
        void            print(void);
        virtual void    setBrlen(double x);
    
    protected:
        void            setTransitionProbabilities(double v);
        double          brlen;
        double*         valsEnd;
        double          vals[16];
};

class FirstDerivatives : public TransitionMatrix {

    public:
                        FirstDerivatives(void);
        void            setBrlen(double x) override;
    
    private:
        void            setFirstDerivatives(double v);
};

class SecondDerivatives : public TransitionMatrix {

    public:
                        SecondDerivatives(void);
        void            setBrlen(double x) override;
    
    private:
        void            setSecondDerivatives(double v);
};

#endif 
