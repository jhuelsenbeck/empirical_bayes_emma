#include <iostream>
#include "Threads.hpp"
#include "Msg.hpp"

ThreadTask::ThreadTask(void) {
}

ThreadTask::~ThreadTask(void) {
}

LikelihoodTask::~LikelihoodTask(void) {
}

#if 1

// Threaded version
ThreadPool::ThreadPool(void) :
    tasksInFlight(0),
    queueHead(0),
    queueTail(0),
    queueSize(0),
    threadCount(std::thread::hardware_concurrency()),
    running(true) {

    if (threadCount < 1)
        threadCount = 1;

    threads = new std::thread[threadCount];
    
    // spawn worker threads
    for (int i = 0; i < threadCount; i++)
        threads[i] = std::thread(&ThreadPool::worker, this);
}

ThreadPool::~ThreadPool(void) {

    // wait for all pending work to complete
    wait();
    
    // signal shutdown
    {
        std::lock_guard<std::mutex> lock(mutex);
        running = false;
    }
    
    // wake all workers so they can exit
    taskAvailable.notify_all();
    
    // join all threads
    if (threads)
        {
        for (int i = 0; i < threadCount; i++)
            threads[i].join();
        delete [] threads;
        }
}

void ThreadPool::pushTask(ThreadTask* task) {

    std::lock_guard<std::mutex> lock(mutex);
    
    // check for queue overflow
    if (queueSize >= queueCapacity)
        Msg::error("ThreadPool queue overflow");
    
    // add to circular buffer (fast modulo via bitmask)
    taskQueue[queueTail] = task;
    queueTail = (queueTail + 1) & queueMask;
    ++queueSize;
    ++tasksInFlight;
    
    // wake one waiting worker
    taskAvailable.notify_one();
}

ThreadTask* ThreadPool::popTask(void) {

    std::unique_lock<std::mutex> lock(mutex);
    
    // wait until there's a task OR we're shutting down
    taskAvailable.wait(lock, [this] {
        return queueSize > 0 || !running;
    });
    
    // check shutdown condition
    if (!running && queueSize == 0)
        return nullptr;
    
    // pop from circular buffer
    ThreadTask* task = taskQueue[queueHead];
    queueHead = (queueHead + 1) & queueMask;
    --queueSize;
    
    return task;
}

void ThreadPool::wait(void) {

    std::unique_lock<std::mutex> lock(mutex);
    
    // block until all tasks are complete (no spinning)
    allComplete.wait(lock, [this] {
        return tasksInFlight == 0;
    });
}

void ThreadPool::worker(void) {
    
    while (running)
        {
        ThreadTask* task = popTask();
        
        if (task)
            {
            // execute outside any lock -- this is where parallelism happens
            try
                {
                task->run();
                }
            catch (const std::exception& e)
                {
                std::cerr << "EXCEPTION in task: " << e.what() << std::endl;
                }
            catch (...)
                {
                std::cerr << "UNKNOWN EXCEPTION in task" << std::endl;
                }
            
            // decrement in-flight count and potentially wake wait()
            // MUST hold mutex to synchronize with wait()
            {
            std::lock_guard<std::mutex> lock(mutex);
            size_t remaining = --tasksInFlight;
            if (remaining == 0)
                allComplete.notify_all();
            }
            }
        // no yield() here -- immediately try to get next task
        }
}

#else

// Serial version (for debugging or single-threaded builds)
ThreadPool::ThreadPool(void):
    queueHead(0),
    queueTail(0),
    queueSize(0),
    threadCount(1),
    tasksInFlight(0),
    running(false),
    threads(nullptr)
{
}

ThreadPool::~ThreadPool(void) {
}

void ThreadPool::pushTask(ThreadTask* task) {

    // execute immediately in serial mode
    task->run();
}

ThreadTask* ThreadPool::popTask(void) {

    return nullptr;
}

void ThreadPool::wait(void) {

    // Nothing to wait for in serial mode - tasks execute synchronously
}

void ThreadPool::worker(void) {
}

#endif
