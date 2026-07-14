#include "TreeCache.hpp"
#include "TreeKey.hpp"



TreeKey::TreeKey(void) : isInitialized(false) {

}

void TreeKey::initialize(TreeCache* cache) {

    if (isInitialized == true)
        return;
        
    TreeCacheMap& tCache = cache->getCache();
    unsigned cnt = 0;
    for (auto& [key,val] : tCache)
        {
        cnt++;
        treeNumbers.insert(std::make_pair(key,cnt));
        }
        
    isInitialized = true;
}

unsigned TreeKey::numberForTreeHash(uint64_t hash) {

    if (isInitialized == false)
        return 0;
    KeyMap::iterator it = treeNumbers.find(hash);
    if (it == treeNumbers.end())
        return 0;
    return it->second;
}
