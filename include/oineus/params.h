#ifndef OINEUS_PARAMS_H
#define OINEUS_PARAMS_H

#include <map>
#include <fstream>
#include <iostream>

#include "log_wrapper.h"

namespace oineus {
    struct Params {

        int n_threads{1};
        int chunk_size{128};
        bool write_dgms{false};
        bool sort_dgms{false};
        bool clearing_opt{true};
        bool acq_rel{false};
        bool print_time{false};
        bool compute_v{false};
        bool compute_u{false};
        bool do_sanity_check{false};
        double elapsed{0.0};
        bool verbose{false};
        spd::level::level_enum spdlog_level {spd::level::level_enum::info};
    };

    inline std::ostream& operator<<(std::ostream& out, const Params& p)
    {
        out << "Params(n_threads = " << p.n_threads;
        out << ", chunk_size = " << p.chunk_size;
        out << ", write_dgms = " << p.write_dgms;
        out << ", sort_dgms = " << p.sort_dgms;
        out << ", clearing_opt = " << p.clearing_opt;
        out << ", acq_rel = " << p.acq_rel;
        out << ", print_time = " << p.print_time;
        out << ", compute_v = " << p.compute_v;
        out << ", compute_u = " << p.compute_u;
        out << ", do_sanity_check = " << p.do_sanity_check;
        out << ", elapsed = " << p.elapsed;
        out << ", verbose = " << p.verbose;
        out << ")";
        return out;
    }

    struct ThreadStats {
        const int thread_id;
        long int n_right_pivots {0};
        long int n_cleared {0};

#ifdef OINEUS_GATHER_ADD_STATS
        using AddStats = std::map<std::pair<size_t, size_t>, size_t>;
        // key: size of pivot column size of right column (the column to which we add pivot)
        AddStats r_column_summand_sizes;
        AddStats v_column_summand_sizes;
#endif

        ThreadStats()
                :thread_id(-1) { }

        ThreadStats(int _thread_id)
                :thread_id(_thread_id) { }
    };

#ifdef OINEUS_GATHER_ADD_STATS
    inline void write_add_stats_file(const std::vector<ThreadStats>& stats)
    {
        ThreadStats::AddStats total_r_stats, total_v_stats;
        for(const auto& s: stats) {
            for(auto[k, v]: s.r_column_summand_sizes)
                total_r_stats[k] += v;
            for(auto[k, v]: s.v_column_summand_sizes)
                total_v_stats[k] += v;
        }

        std::ofstream f_r("add_stats_r.bin", std::ios::binary);

        if (not f_r.good()) {
            std::cerr << "Cannot write column size stats to add_stats_r.bin" << std::endl;
        } else {
//            std::cerr << "writing to add_stats_r.bin, stats size = " << total_r_stats.size() << std::endl;
            for(auto[k, v]: total_r_stats) {
                f_r.write(reinterpret_cast<const char*>(&(k.first)), sizeof(k.first));
                f_r.write(reinterpret_cast<const char*>(&(k.second)), sizeof(k.first));
                f_r.write(reinterpret_cast<const char*>(&v), sizeof(v));
            }

            f_r.close();
        }

        std::ofstream f_v("add_stats_v.bin", std::ios::binary);

        if (not f_v.good()) {
            std::cerr << "Cannot write column size stats to add_stats_v.bin" << std::endl;
        } else {
            for(auto[k, v]: total_v_stats) {
                f_v.write(reinterpret_cast<const char*>(&(k.first)), sizeof(k.first));
                f_v.write(reinterpret_cast<const char*>(&(k.second)), sizeof(k.first));
                f_v.write(reinterpret_cast<const char*>(&v), sizeof(v));
            }

            f_v.close();
        }
    }
#endif

    inline std::ostream& operator<<(std::ostream& out, const oineus::ThreadStats& p)
    {
        out << "Stats(thread_id = " << p.thread_id << ", n_cleared = " << p.n_cleared << ", n_right_pivots = " << p.n_right_pivots <<")";
        return out;
    }
}

#endif //OINEUS_PARAMS_H
