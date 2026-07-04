#ifndef OINEUS_PARAMS_H
#define OINEUS_PARAMS_H

#include <map>
#include <fstream>
#include <iostream>

#include "common_defs.h"

namespace oineus {
    // Working-column data structure used during reduction. The at-rest storage
    // of a column is always a sorted std::vector; this only selects the
    // transient residual representation (see column_repr.h).
    enum class ColumnRepr {
        Set = 0,        // std::set (PHAT A-Set)
        Heap = 1,       // lazy max-heap (PHAT A-Heap)
        Full = 2,       // dense bitset + max-heap (PHAT A-Full)
        BitTree = 3     // hierarchical 64-ary bitset (PHAT A-Bit-Tree)
    };

    inline const char* to_string(ColumnRepr cr)
    {
        switch (cr) {
            case ColumnRepr::Set:        return "Set";
            case ColumnRepr::Heap:       return "Heap";
            case ColumnRepr::Full:       return "Full";
            case ColumnRepr::BitTree:    return "BitTree";
        }
        return "Unknown";
    }

    inline std::ostream& operator<<(std::ostream& out, ColumnRepr cr)
    {
        out << to_string(cr);
        return out;
    }

    // Tri-state request for the apparent-pairs optimization
    // (ReductionParams::use_apparent_pairs); resolution documented there
    enum class ApparentPairs : char {
        Auto = 0,
        On = 1,
        Off = 2
    };

    inline const char* to_string(ApparentPairs ap)
    {
        switch (ap) {
            case ApparentPairs::Auto: return "auto";
            case ApparentPairs::On:   return "on";
            case ApparentPairs::Off:  return "off";
        }
        return "unknown";
    }

    inline std::ostream& operator<<(std::ostream& out, ApparentPairs ap)
    {
        out << to_string(ap);
        return out;
    }

    struct ReductionParams {

        // Rarely-tuned knobs, grouped so the top level stays small
        struct Advanced {
            int chunk_size{128};
            ColumnRepr col_repr{ColumnRepr::BitTree};
            DimVec dims_to_restore_elz;
        };

        int n_threads{1};
        bool use_clearing{true};
        bool compute_v{false};
        bool compute_u{false};
        // Apparent-pairs optimization (Bauer/Ripser): skip building + storing the
        // apparent columns; pre-seed their pivots. Tri-state:
        //   Auto (default) -- ON only in the measured pure-win corner, the fused
        //     parallel R-only homology reduction of a complete cubical grid
        //     (wall ties-or-wins AND lower peak RSS there); OFF everywhere else.
        //   On -- force wherever supported: the fused path for complete
        //     Cubical/Freudenthal filtrations (not subfiltrations); silently
        //     ignored otherwise. Buys the peak-RSS floor at a wall cost on
        //     cohomology and on large Freudenthal grids.
        //   Off -- never.
        // See include/oineus/apparent.h and docs/topics/performance.md.
        ApparentPairs use_apparent_pairs{ApparentPairs::Auto};
        bool sanity_check{false};
        bool verbose{false};
        Advanced advanced;
    };

    // back-compat alias for the historical C++ name
    using Params = ReductionParams;

    inline std::ostream& operator<<(std::ostream& out, const ReductionParams::Advanced& a)
    {
        out << "ReductionParamsAdvanced(chunk_size = " << a.chunk_size;
        out << ", col_repr = " << a.col_repr;
        out << ", dims_to_restore_elz = [";
        for(size_t i = 0; i < a.dims_to_restore_elz.size(); ++i)
            out << (i ? ", " : "") << a.dims_to_restore_elz[i];
        out << "])";
        return out;
    }

    inline bool operator==(const ReductionParams::Advanced& a, const ReductionParams::Advanced& b)
    {
        return a.chunk_size == b.chunk_size
            && a.col_repr == b.col_repr
            && a.dims_to_restore_elz == b.dims_to_restore_elz;
    }

    inline bool operator!=(const ReductionParams::Advanced& a, const ReductionParams::Advanced& b)
    {
        return !(a == b);
    }

    inline std::ostream& operator<<(std::ostream& out, const ReductionParams& p)
    {
        out << "ReductionParams(n_threads = " << p.n_threads;
        out << ", use_clearing = " << p.use_clearing;
        out << ", compute_v = " << p.compute_v;
        out << ", compute_u = " << p.compute_u;
        out << ", use_apparent_pairs = " << p.use_apparent_pairs;
        out << ", sanity_check = " << p.sanity_check;
        out << ", verbose = " << p.verbose;
        out << ", advanced = " << p.advanced;
        out << ")";
        return out;
    }

    inline bool operator==(const ReductionParams& a, const ReductionParams& b)
    {
        return a.n_threads == b.n_threads
            && a.use_clearing == b.use_clearing
            && a.compute_v == b.compute_v
            && a.compute_u == b.compute_u
            && a.use_apparent_pairs == b.use_apparent_pairs
            && a.sanity_check == b.sanity_check
            && a.verbose == b.verbose
            && a.advanced == b.advanced;
    }

    inline bool operator!=(const ReductionParams& a, const ReductionParams& b)
    {
        return !(a == b);
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
