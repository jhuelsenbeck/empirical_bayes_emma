#include <iostream>
#include <limits>
#include <sstream>
#include "Msg.hpp"
#include "BitSet.hpp"

int BitSet::bitsPerUint = sizeof(unsigned) * 8;
unsigned BitSet::maxUInt = std::numeric_limits<unsigned>::max() ;
unsigned BitSet::highBit = (unsigned(1) << (BitSet::bitsPerUint-1));



// default constructor
BitSet::BitSet(void) {

    numElements = 0;
    numUints = 0;
    mask = 0;
    v.clear();
}

// copy constructor
BitSet::BitSet(const BitSet& b) {

    clone(b);
}

// constructor
BitSet::BitSet(size_t n) {

    // calculate size of vector and allocate memory for bitfield
    numElements = n;
    numUints = (numElements / BitSet::bitsPerUint);
    if (numElements % BitSet::bitsPerUint != 0)
        numUints++;
    v.resize(numUints);
    for (size_t i=0; i<numUints; i++)
        v[i] = 0;
    
    // initialize the mask
    if (numElements % BitSet::bitsPerUint != 0)
        {
        mask = (BitSet::highBit >> (numElements % BitSet::bitsPerUint));
        mask--;
        mask ^= BitSet::maxUInt;
        }
    else
        {
        mask = BitSet::maxUInt;
        }
}

BitSet::BitSet(std::vector<bool>& bs) {

    numElements = bs.size();
    numUints = (numElements / BitSet::bitsPerUint);
    if (numElements % BitSet::bitsPerUint != 0)
        numUints++;
    v.resize(numUints);
    for (size_t i=0; i<numUints; i++)
        v[i] = 0;
    
    // initialize the mask
    if (numElements % BitSet::bitsPerUint != 0)
        {
        mask = (BitSet::highBit >> (numElements % BitSet::bitsPerUint));
        mask--;
        mask ^= BitSet::maxUInt;
        }
    else
        {
        mask = BitSet::maxUInt;
        }

    // set all of the bits
    for (size_t i=0; i<numElements; i++)
        {
        if (bs[i] == true)
            set(i);
        else
            unset(i);
        }
}

BitSet::BitSet(size_t n, bool def) {

    // calculate size of vector and allocate memory for bitfield
    numElements = n;
    numUints = (numElements / BitSet::bitsPerUint);
    if (numElements % BitSet::bitsPerUint != 0)
        numUints++;
    v.resize(numUints);
    for (size_t i=0; i<numUints; i++)
        v[i] = 0;
    
    // initialize the mask
    if (numElements % BitSet::bitsPerUint != 0)
        {
        mask = (BitSet::highBit >> (numElements % BitSet::bitsPerUint));
        mask--;
        mask ^= BitSet::maxUInt;
        }
    else
        {
        mask = BitSet::maxUInt;
        }

    // set all of the bits
    for (size_t i=0; i<numElements; i++)
        {
        if (def == true)
            set(i);
        else
            unset(i);
        }
}

// subscript operator
bool BitSet::operator[](size_t idx) {

    if (idx >= numElements)
        Msg::error("Subscript index out-of-range");
    return (( v[idx/BitSet::bitsPerUint] & (BitSet::highBit >> (idx % BitSet::bitsPerUint))) != 0);
}

// subscript operator
bool BitSet::operator[](size_t idx) const {

    if (idx >= numElements)
        Msg::error("Subscript index out-of-range");
    return (( v[idx/BitSet::bitsPerUint] & (BitSet::highBit >> (idx % BitSet::bitsPerUint))) != 0);
}

// assignment operator
BitSet& BitSet::operator=(const BitSet& b) {

    if (this != &b)
        clone(b);
    return *this;
}

// equality operator
bool BitSet::operator==(const BitSet& b) const {

    if (numElements != b.numElements)
        return false;
    for (unsigned long i=0; i<numUints; i++)
        if (v[i] != b.v[i])
            return false;
    return true;
}

// not operator
BitSet operator!(const BitSet& b) {

    BitSet newB(b);
    newB.flip();
    return newB;
}

// inequality operator
bool operator!=(const BitSet& lhs, const BitSet& rhs) {

    return !(lhs == rhs);
}

// less than or equal to operator
bool operator<=(const BitSet& a, const BitSet& b) {

    if (a > b)
        return false;
    else
        return true;
}

// less than operator
bool operator<(const BitSet& lhs, const BitSet& rhs) {

    if (lhs > rhs || lhs == rhs)
        return false;
    else
        return true;
}

// greater than or equal to operator
bool operator>=(const BitSet& lhs, const BitSet& rhs) {

    if (lhs > rhs || lhs == rhs)
        return true;
    else
        return false;
}

// greater than operator
bool BitSet::operator>(const BitSet& b) const {

    size_t x = (numUints < b.numUints) ? numUints : b.numUints;
    for (size_t i=0; i<x; i++)
        {
        if (v[i] > b.v[i])
            return true;
        else if (v[i] < b.v[i])
            return false;
        }
    return false;
}

// AND operator
BitSet BitSet::operator&(const BitSet& a) const {

    if (a.numElements != numElements )
        return BitSet();
    else
        {
        BitSet b(numElements);
        for (size_t i=0; i<numUints; i++)
            b.v[i] = v[i] & a.v[i];
        return b;
        }
}

// OR operator
BitSet BitSet::operator|(const BitSet& a) const {

    if (a.numElements != numElements )
        return BitSet();
    else
        {
        BitSet b(numElements);
        for (size_t i=0; i<numUints; i++)
            b.v[i] = v[i] | a.v[i];
        return b;
        }
}

// XOR operator
BitSet BitSet::operator^(const BitSet& a) const {

    if (a.numElements != numElements )
        return BitSet();
    else
        {
        BitSet b(numElements);
        for (size_t i=0; i<numUints; i++)
            b.v[i] = v[i] ^ a.v[i];
        return b;
        }
}

// unary not
BitSet& BitSet::operator~() {

    flip();
    return *this;
}

// stream operator
std::ostream& operator<<(std::ostream& s, const BitSet& b) {

    s << "[" << b.bitString() << "]";
    return s;
}

// AND equals operator
BitSet& operator&=(BitSet& lhs, const BitSet& rhs) {

    if (lhs.size() == rhs.size())
        {
        for (size_t i=0; i<lhs.getNumUnints(); i++)
            {
            unsigned lvi = lhs.getUintElement(i);
            unsigned rvi = rhs.getUintElement(i);
            lhs.setUintElement(i, lvi &= rvi);
            }
        }
    return lhs;
}

// OR equals operator
BitSet& operator|=(BitSet& lhs, const BitSet& rhs) {

    if (lhs.size() == rhs.size())
        {
        for (size_t i=0; i<lhs.getNumUnints(); i++)
            {
            unsigned lvi = lhs.getUintElement(i);
            unsigned rvi = rhs.getUintElement(i);
            lhs.setUintElement(i, lvi |= rvi);
            }
        }
    return lhs;
}

// XOR equals operator
BitSet& operator^=(BitSet& lhs, const BitSet& rhs) {

    if (lhs.size() == rhs.size())
        {
        for (size_t i=0; i<lhs.getNumUnints(); i++)
            {
            unsigned lvi = lhs.getUintElement(i);
            unsigned rvi = rhs.getUintElement(i);
            lhs.setUintElement(i, lvi ^= rvi);
            }
        }
    return lhs;
}

// return a string with the bitset
std::string BitSet::bitString(void) const {

    std::string str = "";
    for (size_t i=0; i<numElements; i++)
        {
        bool tf = (*this)[i];
        if (tf == true)
            str += "1";
        else
            str += "0";
        }
    return str;
}

// clone function
void BitSet::clone(const BitSet& b) {

    numElements = b.numElements;
    numUints    = b.numUints;
    mask        = b.mask;
    v           = b.v;
}

// erase all bits
void BitSet::clear(void) {

    numElements = 0;
    numUints = 0;
    mask = 0;
    v.clear();
}

// is the bitset of size zero
bool BitSet::empty(void) const {

    return v.empty();
}

// flip the bit at the index
void BitSet::flip(size_t idx) {

    v[idx/BitSet::bitsPerUint] ^= ((BitSet::highBit >> (idx % BitSet::bitsPerUint)));
}

// flip all bits
void BitSet::flip(void) {

    for (size_t i=0; i<numUints; i++)
        v[i] = v[i] ^ BitSet::maxUInt;
    v[numUints-1] &= mask;
}

// position of the first on bit
size_t BitSet::getFirstSetBit(void) const {
    
    unsigned long i;
    for (i=0; i<numUints; i++)
        {
        if (v[i] != 0)
            break;
        }
    if (i == numUints)
        return (size_t)-1;
    int j;
    for (j=0; j<BitSet::bitsPerUint; j++)
        {
        if ((v[i] & (BitSet::highBit >> j)) != 0)
            break;
        }
    return (i * BitSet::bitsPerUint + j);
}

// number of on bits
size_t BitSet::getNumberSetBits(void) const {
    
    size_t count = 0;
    for (size_t i=0; i<numElements; i++)
        {
        if ( (v[i/BitSet::bitsPerUint] & (BitSet::highBit >> (i % BitSet::bitsPerUint))) != 0 )
            count++;
        }
    return count;
}

// insert a bit
void BitSet::insert(size_t pos, bool tf) {

    // check position of insertion
    if (pos > numElements)
        {
        std::cout << "Cannot insert bit at position " << pos << " because there are only " << numElements << " bits" << std::endl;
        return;
        }
        
    // resize, if necessary
    size_t n = numElements + 1;
    size_t nu = (n / BitSet::bitsPerUint);
    if (n % BitSet::bitsPerUint != 0)
        nu++;
    if (nu != numUints)
        v.push_back(0);
    
    numElements = n;
    numUints = nu;
    
    // initialize the mask
    if (numElements % BitSet::bitsPerUint != 0)
        {
        mask = (BitSet::highBit >> (numElements % BitSet::bitsPerUint));
        mask--;
        mask ^= BitSet::maxUInt;
        }
    else
        {
        mask = BitSet::maxUInt;
        }

    // set the bits
    for (size_t i=numElements-1; i>pos; i--)
        {
        if ( (*this)[i-1] == true )
            set(i);
        else
            unset(i);
        }
    if (tf == true)
        set(pos);
    else
        unset(pos);
}

// the number of on bits
bool BitSet::isSet(size_t idx) const {

    return ((v[idx/BitSet::bitsPerUint] & (BitSet::highBit >> (idx % BitSet::bitsPerUint))) != 0);
}

// print in a long format the bitset
void BitSet::print(void) {

    std::string str = "";
    str += "BitSet\n";
    str += "   Size     = " + std::to_string(numElements) + "\n";
    str += "   Capacity = " + std::to_string(numUints*BitSet::bitsPerUint) + "\n   ";
    str += bitString();
    std::cout << str << std::endl;
}

// add a bit to the end of the set
void BitSet::push_back(bool tf) {

    size_t n = numElements + 1;
    size_t nu = (n / BitSet::bitsPerUint);
    if (n % BitSet::bitsPerUint != 0)
        nu++;
    if (nu != numUints)
        v.push_back(0);
    
    numElements = n;
    numUints = nu;
    
    // initialize the mask
    if (numElements % BitSet::bitsPerUint != 0)
        {
        mask = (BitSet::highBit >> (numElements % BitSet::bitsPerUint));
        mask--;
        mask ^= BitSet::maxUInt;
        }
    else
        {
        mask = BitSet::maxUInt;
        }
    if (tf == true)
        set(numElements-1);
    else
        unset(numElements-1);
}

// add a bit to the front of the set
void BitSet::push_front(bool tf) {

    size_t n = numElements + 1;
    size_t nu = (n / BitSet::bitsPerUint);
    if (n % BitSet::bitsPerUint != 0)
        nu++;
    if (nu != numUints)
        v.push_back(0);
    
    numElements = n;
    numUints = nu;
    
    // initialize the mask
    if (numElements % BitSet::bitsPerUint != 0)
        {
        mask = (BitSet::highBit >> (numElements % BitSet::bitsPerUint));
        mask--;
        mask ^= BitSet::maxUInt;
        }
    else
        {
        mask = BitSet::maxUInt;
        }

    for (size_t i=numElements-1; i>=1; i--)
        {
        if ( (*this)[i-1] == true )
            set(i);
        else
            unset(i);
        }
    if (tf == true)
        set(0);
    else
        unset(0);
}

void BitSet::resize(size_t n) {

    // size the vector
    numElements = n;
    numUints = (numElements / BitSet::bitsPerUint);
    if (numElements % BitSet::bitsPerUint != 0)
        numUints++;
    v.resize(numUints);
    for (size_t i=0; i<numUints; i++)
        v[i] = 0;
    
    // initialize the mask
    if (numElements % BitSet::bitsPerUint != 0)
        {
        mask = (BitSet::highBit >> (numElements % BitSet::bitsPerUint));
        mask--;
        mask ^= BitSet::maxUInt;
        }
    else
        {
        mask = BitSet::maxUInt;
        }
}

void BitSet::resize(size_t n, bool tf) {

    resize(n); // this sets all bits to false
    if (tf == true)
        {
        for (size_t i=0; i<numElements; i++)
            set(i);
        }
}

// set all bist to on
void BitSet::set(void) {

    for (size_t i=0; i<numElements; i++)
        v[i/BitSet::bitsPerUint] |= (BitSet::highBit >> (i % BitSet::bitsPerUint));
}

// set a bit to on
void BitSet::set(size_t idx) {

    if (idx >= numElements)
        Msg::error("Subscript index out-of-range");
    v[idx/BitSet::bitsPerUint] |= (BitSet::highBit >> (idx % BitSet::bitsPerUint));
}

// set all bits to 0
void BitSet::unset(void) {

    for (size_t i=0; i<numElements; i++)
        v[i/BitSet::bitsPerUint] &= ((BitSet::highBit >> (i % BitSet::bitsPerUint))^BitSet::maxUInt);
}

// set a bit to 0
void BitSet::unset(size_t idx) {

    if (idx >= numElements)
        Msg::error("Subscript index out-of-range");
    v[idx/BitSet::bitsPerUint] &= ((BitSet::highBit >> (idx % BitSet::bitsPerUint))^BitSet::maxUInt);
}

// check the functionality of the class
void BitSet::test(void) {

    // first check that edge cases are OK
    BitSet x1(10);
    x1.set(3);
    std::cout << x1 << " should be: (0001000000)" << std::endl;
    ~x1;
    std::cout << x1 << " should be: (1110111111)" << std::endl;
    BitSet x2(32);
    x2.set(31);
    std::cout << x2 << " should be: (00000000000000000000000000000001)" << std::endl;
    BitSet x3(33);
    x3.set(32);
    std::cout << x3 << " should be: (000000000000000000000000000000001)" << std::endl;
    
    // check operators
    BitSet x4(10);
    x4.set(4);
    std::cout << (x1 == x4) << " should be 0" << std::endl;
    BitSet x5(33);
    x5.set(32);
    std::cout << (x3 == x5) << " should be 1" << std::endl;
    std::cout << (x1 > x4) << " should be 1" << std::endl;
    std::cout << (x1 | x4) << " should be (0001100000)" << std::endl;
    std::cout << (x1 & x4) << " should be (0000000000)" << std::endl;
    std::cout << (x1 ^ x4) << " should be (0001100000)" << std::endl;
    std::cout << x1[3] << x2[31] << x3[32] << x4[4] << x5[32] << " should be 11111" << std::endl;
    std::cout << x1[0] << x2[14] << x3[30] << x4[8] << x5[29] << " should be 00000" << std::endl;
    
    BitSet x6(1);
    for (int i=0; i<100; i++)
        {
        x6.push_back(true);
        std::cout << x6 << std::endl;
        }

    BitSet x7(1);
    for (int i=0; i<100; i++)
        {
        x7.push_front(true);
        std::cout << x7 << std::endl;
        }
        
    BitSet x8(32);
    x8.insert(1, true);
    std::cout << x8 << std::endl;
    x8.print();

    BitSet x9(32);
    x9.insert(33, true);
    std::cout << x9 << std::endl;
    x9.print();
}
