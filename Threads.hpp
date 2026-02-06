#ifndef ThreadPool_hpp
#define ThreadPool_hpp

#include <thread>
#include <mutex> 
#include <condition_variable>
#include <atomic>



class ThreadTask {

    public:
                                ThreadTask(void);
        virtual                ~ThreadTask(void);
        virtual void            run(void) = 0;
};

class LikelihoodTask : public ThreadTask {

    public: 
        virtual                ~LikelihoodTask(void);
        void                    run(void) override {
                                    result = lnLikelihood();
                                }
        double                  getResult(void) const { return result; }
        
    protected:
        virtual double          lnLikelihood(void) = 0;
        
    private:
        double                  result;
};

class ThreadPool {

    public:
        explicit                ThreadPool(void);
                               ~ThreadPool(void);
        void                    pushTask(ThreadTask* task);
        void                    wait(void);
        int                     getThreadCount(void) const { return threadCount; }

    private:
        void                    worker(void);
        ThreadTask*             popTask(void);
        static constexpr size_t queueCapacity = 1024;           // cache-friendly circular buffer for tasks
        static constexpr size_t queueMask = queueCapacity - 1;  // note the queueCapacity size must be power of 2 for fast modulo via bitwise AND
        std::atomic<size_t>     tasksInFlight;                  // tasks queued + tasks being executed
        size_t                  queueHead;                      // next slot to read from
        size_t                  queueTail;                      // next slot to write to
        size_t                  queueSize;                      // current number of tasks in queue
        std::thread*            threads;
        int                     threadCount;                    // 4 bytes
        std::mutex              mutex;                          // single mutex protects the queue
        std::condition_variable taskAvailable;                  // signaled when queue non-empty
        std::condition_variable allComplete;                    // signaled when tasksInFlight hits 0
        std::atomic<bool>       running;                        // 1 byte
        ThreadTask*             taskQueue[queueCapacity];
};


#endif
