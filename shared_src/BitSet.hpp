#ifndef BitSet_H
#define BitSet_H

#include <string>
#include <vector>

/* The BitSet class implements a dynamic array of bits. The bits are densely packed
   into an array of unsigned integers. This array is a STL vector of unsigned ints. The
   bitset can be dynamically resized by pushing bits to the front or back or bby arbitrarily
   inserting a bit. Bits can also be deleted. */

class BitSet {

    public:
                                BitSet(void);                                           // Default constructor (blank bitset)
                                BitSet(const BitSet& b);                                // Copy constructor
                                BitSet(size_t n);                                       // Constructor setting up an empty bitset of size n
                                BitSet(size_t n, bool def);                             // Constructor setting up an array of size n with all bits set to def
                                BitSet(std::vector<bool>& bs);
        bool                    operator[](size_t idx);                                 // Indexing of bits
        bool                    operator[](size_t idx) const;                           // Indexing of bits  (for const bitset)
        BitSet&                 operator=(const BitSet& b);                             // Assignment operator (deep copy)
        bool                    operator==(const BitSet& b) const;                      // Equality operator
        bool                    operator>(const BitSet& b) const;                       // Larger than operator
        BitSet                  operator&(const BitSet& b) const;                       // Operator &
        BitSet                  operator|(const BitSet& b) const;                       // Operator |
        BitSet                  operator^(const BitSet& b) const;                       // Operator ^
        BitSet&                 operator~();                                            // Unary not
        friend std::ostream&    operator<<(std::ostream& s, const BitSet& b);           // Operator <<
        std::string             bitString(void) const;                                  // Get a string representation of the bitset
        void                    clear(void);                                            // Remove all elements, setting size to zero
        std::vector<unsigned>&  data(void) { return v; }
        size_t                  getFirstSetBit(void) const;                             // Return the index of the first bit that is on (1)
        bool                    empty(void) const;                                      // Whether the bitset is empty (is of size zero)
        void                    flip(size_t idx);                                       // Flip bit at position idx
        void                    flip(void);                                             // Flip all of the bits from 0 to 1 and from 1 to 0
        size_t                  getNumberSetBits(void) const;                           // Return the number of bits that are on
        size_t                  getNumUnints(void) const { return numUints; }           // Return the number of unsigned ints used to store the bits in the set
        unsigned                getUintElement(size_t idx) const { return v[idx]; }     // Return the unsigned ints used to store the bits in the set at location idx
        void                    insert(size_t pos, bool tf);                            // Insert a bit at position pos
        bool                    isSet(size_t idx) const;                                // Is bit at position idx set (on)?
        void                    print(void);                                            // Print a long description of the bitset
        void                    push_back(bool tf);                                     // Add a bit to the end of the vector of bits
        void                    push_front(bool tf);                                    // Add a bit to the front of the vector of bits
        void                    resize(size_t n);                                       // Resize the vector , setting all bits to false
        void                    resize(size_t n, bool tf);                              // Resize the vector and set them all to tf
        void                    set(void);                                              // set all bits to on (1)
        void                    set(size_t idx);                                        // set bit at position idx to on (1)
        void                    setUintElement(size_t idx, unsigned x) { v[idx] = x; }  // Set the unsigned ints used to store the bits in the set at location idx
        size_t                  size(void) const { return numElements; }                // Return the number of bits in the set
        void                    test(void);                                             // Run a series of tests on the class
        void                    unset(void);                                            // set all bits to off (0)
        void                    unset(size_t idx);                                      // set bit at position idx to off (0)

    private:
        void                    clone(const BitSet& b);
        static int              bitsPerUint;                                            // The number of bits per unsigned integer
        static unsigned         maxUInt;                                                // The maximum value of an unsigned integer
        static unsigned         highBit;                                                // An unsigned integer with only the highest bit on
        size_t                  numElements;                                            // The number of bits
        size_t                  numUints;                                               // The number of unsigned integers needed to store the bits
        unsigned                mask;                                                   // A mask for the last unsigned integer in the array
        std::vector<unsigned>   v;                                                      // The array of Uints that store the bits
};

        BitSet                  operator!(const BitSet& b);                             // Operator !
        BitSet&                 operator&=(BitSet& lhs, const BitSet& rhs);             // Operator &=
        BitSet&                 operator|=(BitSet& lhs, const BitSet& rhs);             // Operator |=
        BitSet&                 operator^=(BitSet& lhs, const BitSet& rhs);             // Operator ^=
        bool                    operator!=(const BitSet& lhs, const BitSet& rhs);       // Inequality operator
        bool                    operator< (const BitSet& lhs, const BitSet& rhs);       // Operator <
        bool                    operator<=(const BitSet& lhs, const BitSet& rhs);       // Operator <=
        bool                    operator>=(const BitSet& lhs, const BitSet& rhs);       // Operator >=

struct CompBitSet {

    bool operator()(const BitSet& n1, const BitSet& n2) const {
        
        if (n1.size() == n2.size())
            {
            for (size_t i=0; i<n1.size(); i++)
                {
                bool v1 = n1.isSet(i);
                bool v2 = n2.isSet(i);
                if (v1 != v2)
                    {
                    if (v1 == true && v2 == false)
                        return true;
                    else
                        return false;
                    }
                }
            }
        else if (n1.size() > n2.size())
            return true;
        else
            return false;
        return false;
        }

    bool operator()(const BitSet* n1, const BitSet* n2) const {
        
        if (n1->size() == n2->size())
            {
            for (size_t i=0; i<n1->size(); i++)
                {
                bool v1 = n1->isSet(i);
                bool v2 = n2->isSet(i);
                if (v1 != v2)
                    {
                    if (v1 == true && v2 == false)
                        return true;
                    else
                        return false;
                    }
                }
            }
        else if (n1->size() > n2->size())
            return true;
        else
            return false;
        return false;
        }
};

#endif
