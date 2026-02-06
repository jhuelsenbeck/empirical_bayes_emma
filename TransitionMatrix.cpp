#include <cmath>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include "TransitionMatrix.hpp"



TransitionMatrix::TransitionMatrix(void) {

    valsEnd = vals + 16;

    brlen = 0.0;
    for (double* p=vals; p != valsEnd; p++)
        *p = 0.0;
    for (double* p=vals; p < valsEnd; p+=5)
        *p = 1.0;
}

void TransitionMatrix::print(void) {

    for (size_t i=0; i<4; i++)
        {
        for (size_t j=0; j<4; j++)
            {
            std::cout << std::fixed << std::setprecision(6);
            std::cout << (*this)(i,j) << " ";
            }
        std::cout << std::endl;
        }
}

void TransitionMatrix::setBrlen(double x) {

    brlen = x;
    setTransitionProbabilities(brlen);
}
    
void TransitionMatrix::setTransitionProbabilities(double v) {

    double expX = std::exp(-(4.0/3.0) * v);
    double pChange = 0.25 - 0.25 * expX;
    double pSame   = 0.25 + 0.75 * expX;
    
    double* p = vals;
    // Row 0
    *p++ = pSame;   *p++ = pChange; *p++ = pChange; *p++ = pChange;
    // Row 1  
    *p++ = pChange; *p++ = pSame;   *p++ = pChange; *p++ = pChange;
    // Row 2
    *p++ = pChange; *p++ = pChange; *p++ = pSame;   *p++ = pChange;
    // Row 3
    *p++ = pChange; *p++ = pChange; *p++ = pChange; *p   = pSame;
}

FirstDerivatives::FirstDerivatives(void) : TransitionMatrix() {

}

void FirstDerivatives::setBrlen(double x) {

    brlen = x;
    setFirstDerivatives(brlen);
}
    
void FirstDerivatives::setFirstDerivatives(double v) {

    double expX = std::exp(-(4.0/3.0) * v);
    double pChange = (1.0/3.0) * expX;
    double pSame   = -expX;
    
    double* p = vals;
    // Row 0
    *p++ = pSame;   *p++ = pChange; *p++ = pChange; *p++ = pChange;
    // Row 1  
    *p++ = pChange; *p++ = pSame;   *p++ = pChange; *p++ = pChange;
    // Row 2
    *p++ = pChange; *p++ = pChange; *p++ = pSame;   *p++ = pChange;
    // Row 3
    *p++ = pChange; *p++ = pChange; *p++ = pChange; *p   = pSame;
}

SecondDerivatives::SecondDerivatives(void) : TransitionMatrix() {

}

void SecondDerivatives::setBrlen(double x) {

    brlen = x;
    setSecondDerivatives(brlen);
}
    
void SecondDerivatives::setSecondDerivatives(double v) {

    double expX = std::exp(-(4.0/3.0) * v);
    double pChange = -(4.0/9.0) * expX;
    double pSame   = (4.0 / 3.0) * expX;
    
    double* p = vals;
    // Row 0
    *p++ = pSame;   *p++ = pChange; *p++ = pChange; *p++ = pChange;
    // Row 1  
    *p++ = pChange; *p++ = pSame;   *p++ = pChange; *p++ = pChange;
    // Row 2
    *p++ = pChange; *p++ = pChange; *p++ = pSame;   *p++ = pChange;
    // Row 3
    *p++ = pChange; *p++ = pChange; *p++ = pChange; *p   = pSame;
}
