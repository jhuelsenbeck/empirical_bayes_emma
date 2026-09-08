#include <sstream>
#include "RandomVariable.hpp"



RandomVariable::RandomVariable(void) : uniformDist(0.0, 1.0), initialSeed(0) {

    initializeFromRandomDevice();
}

RandomVariable::RandomVariable(uint32_t seed) : uniformDist(0.0, 1.0), initialSeed(seed) {

    if (seed == 0)
        initializeFromRandomDevice();
    else
        rng.seed(seed);
}

void RandomVariable::initializeFromRandomDevice(void) {

    std::random_device rd;
    
    // use multiple random_device calls for better entropy
    std::seed_seq seq{ rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd() };
    rng.seed(seq);
    
    // store a representative seed for logging (first value from rd)
    initialSeed = rd();
}

double RandomVariable::uniformRv(void) {

    return uniformDist(rng);
}

void RandomVariable::setSeed(uint32_t seed) {

    initialSeed = seed;
    if (seed == 0)
        initializeFromRandomDevice();
    else
        rng.seed(seed);
}
