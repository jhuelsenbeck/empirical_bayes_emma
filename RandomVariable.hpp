#ifndef RandomVariable_hpp
#define RandomVariable_hpp

#include <cstdint>
#include <random>
#include <string>



class RandomVariable {

    public:
                                                RandomVariable(void);
        explicit                                RandomVariable(uint32_t seed);
                                                RandomVariable(const RandomVariable&) = delete;
                                                RandomVariable(RandomVariable&&) = default;
                                               ~RandomVariable(void) = default;
        RandomVariable&                         operator=(const RandomVariable&) = delete;
        std::mt19937&                           getGenerator(void) { return rng; }
        uint32_t                                getSeed(void) const { return initialSeed; }
        void                                    setSeed(uint32_t seed);
        double                                  uniformRv(void); // [0, 1)

    private:
        void                                    initializeFromRandomDevice(void);
        std::mt19937                            rng;
        std::uniform_real_distribution<double>  uniformDist;
        uint32_t                                initialSeed;
};

#endif
