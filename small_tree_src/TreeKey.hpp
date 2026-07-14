#ifndef TreeKey_hpp
#define TreeKey_hpp

#include <unordered_map>
class TreeCache;
typedef std::unordered_map<uint64_t,unsigned> KeyMap;


class TreeKey {

    public:
        static TreeKey&     treeKey(void) 
                                {
                                static thread_local TreeKey key;
                                return key;
                                }
        void                initialize(TreeCache* cache);
        unsigned            numberForTreeHash(uint64_t hash);
    
    private:
                            TreeKey(void);
                            TreeKey(const TreeKey& tp) = delete;
        bool                isInitialized;
        KeyMap              treeNumbers;
};

#endif
