#pragma once

#include <iostream>
#include <atomic>
#include <vector>
#include <string>
#include <thread>
#include <pthread.h>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <chrono>
#include <stdlib.h>

namespace oineus {
    template<class T>
    struct MemoryReclaim {

        using EpochCounter = int;

        int n_threads_;

        const int thread_id_;

        std::vector<T*> to_retire_, retired_, to_delete_;

        bool even_epoch_ {false};

        std::atomic<EpochCounter>& counter_;

        MemoryReclaim(int _n_threads, std::atomic<int>& _epoch_counter, int _thread_id)
                :
                n_threads_(_n_threads),
                thread_id_(_thread_id),
                counter_(_epoch_counter) { }

        // dtor must be called only after all threads finish!
        ~MemoryReclaim()
        {
            for(T* p: to_delete_)
                delete p;

            for(T* p: retired_)
                delete p;

            for(T* p: to_retire_)
                delete p;
        }

        bool is_even_epoch(int counter) const
        {
            return (counter / n_threads_) % 2 == 0;
        }

        void retire(T*& ptr)
        {
            if (ptr) to_retire_.push_back(ptr);
            ptr = nullptr;
        }

        void quiescent()
        {
            if (even_epoch_ != is_even_epoch(counter_)) {

                ++counter_;

                even_epoch_ = not even_epoch_;

                for(T* p: to_delete_)
                    delete p;

                retired_.swap(to_delete_);
                to_retire_.swap(retired_);
                to_retire_.clear();
            }
        }
    };
} // namespace oineus
