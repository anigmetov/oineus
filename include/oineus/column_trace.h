#pragma once

// Compile-guarded column-lifetime tracing for the fused R-only parallel
// reduction (Direction A "oracle ceiling" experiment). Everything here is
// compiled ONLY under -DOINEUS_COLUMN_TRACE; the default build never sees it.
//
// Usage: the fused R-only orchestrator (reduce_from_filtration_fused,
// compute_v == false) records per-column build sizes, arms g_column_trace
// around the reduction core, and dumps one binary record per column to the
// path in $OINEUS_COLUMN_TRACE_FILE. The reducer hooks in parallel_reduction /
// update_column record touch/free events against a single global tick
// counter. One traced reduce at a time (global context, experiment-only).
//
// Concurrency caveat: ticks are taken with a relaxed fetch_add and events on
// different columns race benignly, so the recovered event order is only
// APPROXIMATE under concurrency -- good enough for high-water-mark replay,
// not for exact happens-before reconstruction.

#ifdef OINEUS_COLUMN_TRACE

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>
#include <algorithm>

namespace oineus {

// flags (bitmask)
inline constexpr uint32_t CT_APPARENT_NULL = 1; // never materialized (apparent-null slot)
inline constexpr uint32_t CT_CLEARED = 2;       // slot nulled by clearing
inline constexpr uint32_t CT_ZEROED = 4;        // slot nulled after reducing to zero

struct ColumnTraceRec {
    std::atomic<uint32_t> nnz_build{0};   // nnz at build (would-be nnz for apparent-null)
    std::atomic<uint32_t> flags{0};
    std::atomic<uint32_t> n_touches{0};
    std::atomic<uint64_t> t_build{0};     // 0 == never
    std::atomic<uint64_t> t_first_touch{0};
    std::atomic<uint64_t> t_last_touch{0};
    std::atomic<uint64_t> t_free{0};      // 0 == survived to end of reduce
};

struct ColumnTraceCtx {
    std::vector<ColumnTraceRec> recs;
    std::atomic<uint64_t> clock{1}; // ticks start at 1 so 0 means "never"

    explicit ColumnTraceCtx(size_t n) : recs(n) {}

    uint64_t tick() { return clock.fetch_add(1, std::memory_order_relaxed); }

    void record_build(size_t idx, uint32_t nnz, uint32_t flag_bits)
    {
        auto& r = recs[idx];
        r.nnz_build.store(nnz, std::memory_order_relaxed);
        r.t_build.store(tick(), std::memory_order_relaxed);
        if (flag_bits)
            r.flags.fetch_or(flag_bits, std::memory_order_relaxed);
    }

    void bump_last_(ColumnTraceRec& r, uint64_t t)
    {
        uint64_t prev = r.t_last_touch.load(std::memory_order_relaxed);
        while (prev < t
                && !r.t_last_touch.compare_exchange_weak(prev, t, std::memory_order_relaxed)) {}
    }

    void record_touch(size_t idx)
    {
        const uint64_t t = tick();
        auto& r = recs[idx];
        uint64_t expected = 0;
        r.t_first_touch.compare_exchange_strong(expected, t, std::memory_order_relaxed);
        bump_last_(r, t);
        r.n_touches.fetch_add(1, std::memory_order_relaxed);
    }

    void record_free(size_t idx, uint32_t flag_bits)
    {
        const uint64_t t = tick();
        auto& r = recs[idx];
        r.t_free.store(t, std::memory_order_relaxed);
        r.flags.fetch_or(flag_bits, std::memory_order_relaxed);
        // freeing implies the column was live at this tick
        bump_last_(r, t);
    }

    // Dump format (native-endian, in practice little-endian):
    //   8 bytes  magic "OINCTRC1"
    //   u64      n_cols
    //   u64      final_clock
    //   n_cols records of 7 x u32:
    //     nnz_build, flags, n_touches, t_build, t_first_touch, t_last_touch, t_free
    //   (t_* == 0 means "never happened"; t_free == 0 means freed at end of run)
    // Reader: benchmarks/analyze_column_trace.py
    bool dump(const char* path) const
    {
        const uint64_t final_clock = clock.load(std::memory_order_relaxed);
        if (final_clock >= UINT32_MAX) {
            std::fprintf(stderr, "column trace: clock overflow (%llu ticks), not dumping\n",
                    static_cast<unsigned long long>(final_clock));
            return false;
        }
        std::FILE* f = std::fopen(path, "wb");
        if (f == nullptr) {
            std::fprintf(stderr, "column trace: cannot open %s for writing\n", path);
            return false;
        }
        const char magic[8] = {'O', 'I', 'N', 'C', 'T', 'R', 'C', '1'};
        const uint64_t n = recs.size();
        std::fwrite(magic, 1, 8, f);
        std::fwrite(&n, 8, 1, f);
        std::fwrite(&final_clock, 8, 1, f);
        const size_t chunk = size_t(1) << 20;
        std::vector<uint32_t> buf;
        buf.reserve(7 * chunk);
        for(size_t begin = 0; begin < recs.size(); begin += chunk) {
            const size_t end = std::min(recs.size(), begin + chunk);
            buf.clear();
            for(size_t i = begin; i < end; ++i) {
                const auto& r = recs[i];
                buf.push_back(r.nnz_build.load(std::memory_order_relaxed));
                buf.push_back(r.flags.load(std::memory_order_relaxed));
                buf.push_back(r.n_touches.load(std::memory_order_relaxed));
                buf.push_back(static_cast<uint32_t>(r.t_build.load(std::memory_order_relaxed)));
                buf.push_back(static_cast<uint32_t>(r.t_first_touch.load(std::memory_order_relaxed)));
                buf.push_back(static_cast<uint32_t>(r.t_last_touch.load(std::memory_order_relaxed)));
                buf.push_back(static_cast<uint32_t>(r.t_free.load(std::memory_order_relaxed)));
            }
            std::fwrite(buf.data(), sizeof(uint32_t), buf.size(), f);
        }
        std::fclose(f);
        std::fprintf(stderr, "column trace: dumped %llu records to %s (final clock %llu)\n",
                static_cast<unsigned long long>(n), path,
                static_cast<unsigned long long>(final_clock));
        return true;
    }
};

// Armed by the fused R-only orchestrator around the reduction core; read by
// the touch/free hooks. nullptr == tracing off.
inline ColumnTraceCtx* g_column_trace = nullptr;

} // namespace oineus

#endif // OINEUS_COLUMN_TRACE
