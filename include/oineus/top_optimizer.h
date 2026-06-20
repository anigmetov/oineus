#pragma once

#include <vector>
#include <utility>
#include <type_traits>

#include "filtration.h"
#include "inclusion_filtration.h"
#include "decomposition.h"
//#include "loss.h"

namespace oineus {

// Pick the U-computation strategy used by the crit-sets backward in
// oineus.diff. Exposed to Python via nanobind.
//   Auto          -- production default; resolves to RowPartial today.
//   RowPartial    -- parallel V-only with restore_elz (cheap), then a
//                    row-form partial U pass via Decomposition::
//                    compute_partial_u_rows over only the rows the
//                    walker reads. Falls back to compute_full_u_rows
//                    when the requested row count exceeds a fraction
//                    of the dim's matrix-column range.
//   LegacyInBand  -- clearing off, U built in-band during reduction
//                    (serial). Available as a control / cross-check.
enum class UStrategy { Auto, RowPartial, LegacyInBand };

inline std::ostream& operator<<(std::ostream& out, UStrategy s)
{
    switch (s) {
        case UStrategy::Auto:         out << "UStrategy::Auto"; break;
        case UStrategy::RowPartial:   out << "UStrategy::RowPartial"; break;
        case UStrategy::LegacyInBand: out << "UStrategy::LegacyInBand"; break;
    }
    return out;
}

struct ComputeFlags {
    bool compute_cohomology {false};
    bool compute_homology_u {false};
    bool compute_cohomology_u {false};
};

inline std::ostream& operator<<(std::ostream& out, const ComputeFlags& f)
{
    out << "ComputeFlags(compute_cohomology = " << (f.compute_cohomology ? "True" : "False");
    out << ", compute_homology_u = " << (f.compute_homology_u ? "True" : "False");
    out << ", compute_cohomology_u = " << (f.compute_cohomology_u ? "True)" : "False)");
    return out;
}

template<class Cell_, class Real_>
class TopologyOptimizer {
public:

    static_assert(std::is_floating_point_v<Real_>, "Real_ must be floating point type");

    using Fil = Filtration<Cell_, Real_>;
    using Cell = CellWithValue<Cell_, Real_>;
    using Real = typename Cell::Real;
    using Int = typename Cell::Int;
    using BoundaryMatrix = typename VRUDecomposition<Int>::MatrixData;
    using Indices = std::vector<Int>;
    using Values = std::vector<Real>;
    using DgmTarget = std::unordered_map<DgmPoint<Real>, DgmPoint<Real>>;
    using CriticalSet = std::pair<Real, Indices>;
    using CriticalSets = std::vector<CriticalSet>;

    using Decomposition = VRUDecomposition<Int>;
    using Dgms = Diagrams<Real>;
    using Dgm = typename Dgms::Dgm;

    struct SimplexTarget {
        Real current_value;
        Real target_value;
        bool is_positive;

        bool increase_birth(bool negate) const
        {
            if (not is_positive)
                return false;
            if (negate)
                return target_value < current_value;
            else
                return target_value > current_value;
        }

        bool decrease_birth(bool negate) const
        {
            if (not is_positive)
                return false;
            if (negate)
                return target_value > current_value;
            else
                return target_value < current_value;
        }

        bool increase_death(bool negate) const
        {
            if (is_positive)
                return false;
            if (negate)
                return target_value < current_value;
            else
                return target_value > current_value;
        }

        bool decrease_death(bool negate) const
        {
            if (is_positive)
                return false;
            if (negate)
                return target_value > current_value;
            else
                return target_value < current_value;
        }
    };

    using Target = std::unordered_map<size_t, SimplexTarget>;

    struct IndicesValues {
        Indices indices;
        Values values;

        void push_back(size_t i, Real v)
        {
            indices.push_back(i);
            values.push_back(v);
        }

        void emplace_back(size_t i, Real v)
        {
            indices.emplace_back(i);
            values.emplace_back(v);
        }

        friend std::ostream& operator<<(std::ostream& out, const IndicesValues& iv)
        {
            out << "IndicesValue(indices=";
            out << container_to_string(iv.indices);
            out << ", values=";
            out << container_to_string(iv.values);
            return out;
        }

        bool operator==(const IndicesValues& other) const
        {
            return indices == other.indices && values == other.values;
        }

        bool operator!=(const IndicesValues& other) const
        {
            return !(*this == other);
        }
    };

    // TopologyOptimizer(const Fil& fil)
    //         :
    //         negate_(fil.negate()),
    //         fil_(fil),
    //         decmp_hom_(fil, false),
    //         decmp_coh_(fil, true)
    // {
    //     params_hom_.compute_v = true;
    //     params_coh_.compute_v = true;
    //     params_hom_.compute_u = true;
    //     params_coh_.compute_u = true;
    //     params_hom_.clearing_opt = false;
    //     params_coh_.clearing_opt = false;
    // }
    //
    // TopologyOptimizer(const Fil& fil, const ComputeFlags& hints)
    //         :
    //         decmp_hom_(fil, false),
    //         decmp_coh_(fil, true),
    //         fil_(fil),
    //         negate_(fil.negate())
    // {
    //     params_hom_.compute_u = hints.compute_homology_u;
    //     params_coh_.compute_u = hints.compute_cohomology_u;
    //     params_hom_.clearing_opt = false;
    //     params_coh_.clearing_opt = false;
    // }

    // Recipe is decided here at construction time and stays fixed
    // for the lifetime of the optimizer (one autograd backward).
    //
    //   !with_crit_sets:                  R only, parallel + clearing.
    //   with_crit_sets, !LegacyInBand:    R + V, parallel + clearing,
    //                                     restore_elz in dims_to_restore_elz.
    //   with_crit_sets, LegacyInBand:     R + V + U, serial + clearing off.
    TopologyOptimizer(const Fil& fil,
                      bool with_crit_sets = true,
                      DimVec dims_to_restore_elz = DimVec(),
                      int n_threads = 1,
                      UStrategy u_strategy = UStrategy::Auto)
            :
            negate_(fil.negate()),
            fil_(fil),
            boundary_data_(fil_.boundary_matrix(n_threads)),
            // decmp_hom_, decmp_coh_ default-constructed (empty); they
            // are materialized from boundary_data_ on the first
            // ensure_*_built / ensure_*_reduced call.
            with_crit_sets_(with_crit_sets),
            n_threads_(n_threads),
            dims_to_restore_elz_(std::move(dims_to_restore_elz)),
            u_strategy_(u_strategy)
    {
        const bool legacy_in_band =
            with_crit_sets_ and u_strategy_ == UStrategy::LegacyInBand;

        // If the caller didn't pin which dims to restore ELZ in, default
        // to "all dims" for crit-sets. Otherwise downstream partial-U
        // calls would throw because is_elz_in_dim_ has nothing flipped on.
        if (with_crit_sets_ and not legacy_in_band and dims_to_restore_elz_.empty()) {
            for (dim_type d = 0;
                 d <= fil_.max_dim(); ++d) {
                dims_to_restore_elz_.push_back(d);
            }
        }

        // Mirror constructor inputs into both params; reduction
        // drivers read from Params.
        params_hom_.n_threads = legacy_in_band ? 1 : n_threads_;
        params_coh_.n_threads = legacy_in_band ? 1 : n_threads_;
        // ELZ restoration only happens when V is being built, so the
        // dgm-loss branch below clears this back to empty.
        params_hom_.dims_to_restore_elz = dims_to_restore_elz_;
        params_coh_.dims_to_restore_elz = dims_to_restore_elz_;

        if (not with_crit_sets_) {
            // dgm-loss: only the pairing in R is needed.
            params_hom_.compute_v = false;
            params_coh_.compute_v = false;
            params_hom_.compute_u = false;
            params_coh_.compute_u = false;
            params_hom_.clearing_opt = true;
            params_coh_.clearing_opt = true;
            params_hom_.dims_to_restore_elz.clear();
            params_coh_.dims_to_restore_elz.clear();
        } else if (legacy_in_band) {
            // In-band U: clearing off, serial. The forward already
            // builds U during reduction; ensure_has_u_* will be a no-op.
            params_hom_.compute_v = true;
            params_coh_.compute_v = true;
            params_hom_.compute_u = true;
            params_coh_.compute_u = true;
            params_hom_.clearing_opt = false;
            params_coh_.clearing_opt = false;
        } else {
            // crit-sets default: V is built; U is computed on demand
            // via ensure_has_u_* from a known-ELZ V.
            params_hom_.compute_v = true;
            params_coh_.compute_v = true;
            params_hom_.compute_u = false;
            params_coh_.compute_u = false;
            params_hom_.clearing_opt = true;
            params_coh_.clearing_opt = true;
        }
    }

    // // Tracks how each side was last reduced (or that it has not been).
    // // matrix_summary() returns this for diagnostics + benchmarks; the
    // // ensure_reduced_* helpers also use it to skip work.
    // struct SideStatus {
    //     bool is_reduced {false};
    //     bool has_v {false};
    //     bool has_u {false};
    //     bool clearing_opt_used {false};
    //
    //     friend std::ostream& operator<<(std::ostream& out, const SideStatus& s)
    //     {
    //         out << "SideStatus(is_reduced=" << (s.is_reduced ? "true" : "false")
    //             << ", has_v=" << (s.has_v ? "true" : "false")
    //             << ", has_u=" << (s.has_u ? "true" : "false")
    //             << ", clearing_opt_used=" << (s.clearing_opt_used ? "true" : "false") << ")";
    //         return out;
    //     }
    // };
    //
    // SideStatus side_status(const Decomposition& dcmp, const Params& params) const
    // {
    //     SideStatus s;
    //     s.is_reduced = dcmp.is_reduced;
    //     s.has_v = dcmp.is_reduced and params.compute_v;
    //     s.has_u = dcmp.has_matrix_u();
    //     s.clearing_opt_used = params.clearing_opt;
    //     return s;
    // }
    //
    // SideStatus hom_status() const { return side_status(decmp_hom_, params_hom_); }
    // SideStatus coh_status() const { return side_status(decmp_coh_, params_coh_); }

    // ---------------------------------------------------------------
    // New per-backward API. The optimizer lives for one autograd call;
    // params_hom_/_coh_ were set at construction; idempotency is just
    // is_reduced.

    // Materialize decmp_hom_ from the cached boundary matrix. Idempotent.
    // Safe to call from any context, including before the first reduction.
    void ensure_hom_built()
    {
        if (decmp_hom_built_) return;
        decmp_hom_ = Decomposition(boundary_data_,
                                   fil_.dims_first(), fil_.dims_last(),
                                   /*dualize=*/false, n_threads_);
        decmp_hom_built_ = true;
    }

    void ensure_coh_built()
    {
        if (decmp_coh_built_) return;
        decmp_coh_ = Decomposition(boundary_data_,
                                   fil_.dims_first(), fil_.dims_last(),
                                   /*dualize=*/true, n_threads_);
        decmp_coh_built_ = true;
    }

    bool is_hom_built() const { return decmp_hom_built_; }
    bool is_coh_built() const { return decmp_coh_built_; }

    // Build + reduce in one shot via the fused path, which builds the working
    // column array straight from the cached boundary (no d_data, no prepare-copy)
    // and, for the parallel crit-sets RV case, keeps the reduced RVColumns instead
    // of copying R/V back (read via r_low/r_is_zero/v_col). The read sites'
    // accessors fall back to at-rest data, so the serial / diagram-loss / classic
    // cases (handled inside reduce_from_boundary_fused) work unchanged. If a caller
    // already built the decomposition classically (ensure_*_built), we reduce that
    // in place instead, so an externally-modified build is not discarded.
    void ensure_hom_reduced()
    {
        if (decmp_hom_built_) {
            if (not decmp_hom_.is_reduced)
                decmp_hom_.reduce(params_hom_);
            return;
        }
        decmp_hom_ = Decomposition::reduce_from_boundary_fused(
                boundary_data_, fil_.dims_first(), fil_.dims_last(),
                /*dualize=*/false, params_hom_, /*keep_working=*/true);
        decmp_hom_built_ = true;
    }

    void ensure_coh_reduced()
    {
        if (decmp_coh_built_) {
            if (not decmp_coh_.is_reduced)
                decmp_coh_.reduce(params_coh_);
            return;
        }
        decmp_coh_ = Decomposition::reduce_from_boundary_fused(
                boundary_data_, fil_.dims_first(), fil_.dims_last(),
                /*dualize=*/true, params_coh_, /*keep_working=*/true);
        decmp_coh_built_ = true;
    }

    // Return a reduced decomposition that yields the persistence pairing.
    // The diagram is identical from either side, so reuse whichever side is
    // already reduced; if neither is, fall back to homology. The
    // performance choice of which side to reduce first for the pairing is
    // the caller's (e.g. the Python kind policy reduces cohomology for VR
    // before calling); this only guarantees that some reduced side exists.
    Decomposition& ensure_pairing_reduced()
    {
        if (decmp_hom_.is_reduced)
            return decmp_hom_;
        if (decmp_coh_.is_reduced)
            return decmp_coh_;
        ensure_hom_reduced();
        return decmp_hom_;
    }

private:
    // Find the geometric dim block that owns the given filtration
    // index by walking dim_first / dim_last (filtration layout).
    // Returns -1 if not found.
    static dim_type _find_geom_dim(const Decomposition& dcmp, size_t fil_idx)
    {
        for (size_t d = 0; d < dcmp.dim_first.size(); ++d) {
            if (static_cast<size_t>(dcmp.dim_first[d]) <= fil_idx and
                fil_idx <= static_cast<size_t>(dcmp.dim_last[d]))
                return static_cast<dim_type>(d);
        }
        return -1;
    }

public:
    // Make U-row data available on the hom side over the rows the
    // crit-set walker will read for death moves. rows_fil are
    // filtration indices (== matrix indices on hom). The internal
    // dim block is inferred from the first row index, since on the
    // hom side a death simplex of an H_k pair is (k+1)-dim, not k.
    // For LegacyInBand U was already built in-band by the forward;
    // this is a no-op.
    void ensure_has_u_hom(dim_type /*dim*/, Indices rows_fil, Values bounds)
    {
        ensure_hom_reduced();
        if (u_strategy_ == UStrategy::LegacyInBand) return;
        if (rows_fil.empty()) return;
        if (rows_fil.size() != bounds.size())
            throw std::runtime_error("ensure_has_u_hom: rows/bounds size mismatch");

        // The geometric dim of the death simplex on the hom side is
        // (diagram_dim + 1). Infer it from the first row's
        // filtration index (== matrix index on hom).
        const auto geom_dim = _find_geom_dim(decmp_hom_, static_cast<size_t>(rows_fil[0]));
        if (geom_dim < 0)
            throw std::runtime_error("ensure_has_u_hom: row index out of range");

        auto value_at = [this](size_t midx) -> Real {
            return fil_.get_cell_value(
                fil_.index_in_filtration(midx, /*dualize=*/false));
        };

        // For partial-vs-full sizing we want the count of cells in
        // this dim block on the matrix layout.
        const auto _dim = decmp_hom_._dim_from_dim(geom_dim);
        const size_t dim_size = decmp_hom_._dim_last[_dim]
                              - decmp_hom_._dim_first[_dim] + 1;
        if (4 * rows_fil.size() > 3 * dim_size) {
            decmp_hom_.template compute_full_u_rows<Real>(geom_dim, value_at,
                                           static_cast<size_t>(n_threads_));
            return;
        }

        // Auto/RowPartial: V^T U^T = I. Hom-side U is read for
        // increase_death; on non-negate the walker truncates from
        // above. negate flips the direction.
        std::vector<size_t> rows_sz(rows_fil.begin(), rows_fil.end());
        if (negate_) {
            decmp_hom_.compute_partial_u_rows(rows_sz, bounds, geom_dim,
                value_at,
                [](Real a, Real b) { return a < b; },
                static_cast<size_t>(n_threads_));
        } else {
            decmp_hom_.compute_partial_u_rows(rows_sz, bounds, geom_dim,
                value_at,
                [](Real a, Real b) { return a > b; },
                static_cast<size_t>(n_threads_));
        }
    }

    // Coh-side U for birth moves. rows_fil are filtration indices;
    // we convert to matrix indices (fil_size - 1 - i) before calling
    // the row solver. The internal dim block is inferred from the
    // first matrix-layout row index.
    void ensure_has_u_coh(dim_type /*dim*/, Indices rows_fil, Values bounds)
    {
        ensure_coh_reduced();
        if (u_strategy_ == UStrategy::LegacyInBand) return;
        if (rows_fil.empty()) return;
        if (rows_fil.size() != bounds.size())
            throw std::runtime_error("ensure_has_u_coh: rows/bounds size mismatch");

        const size_t n = fil_.size();
        std::vector<size_t> rows_mat;
        rows_mat.reserve(rows_fil.size());
        for (auto r : rows_fil) rows_mat.push_back(n - 1 - static_cast<size_t>(r));

        // The geometric dim of the birth simplex on the coh side is
        // the diagram dim itself. Infer from the first row's
        // filtration index (rows_fil[0]).
        const auto geom_dim = _find_geom_dim(decmp_coh_, static_cast<size_t>(rows_fil[0]));
        if (geom_dim < 0)
            throw std::runtime_error("ensure_has_u_coh: row index out of range");

        auto value_at = [this](size_t midx) -> Real {
            return fil_.get_cell_value(
                fil_.index_in_filtration(midx, /*dualize=*/true));
        };

        const auto _dim = decmp_coh_._dim_from_dim(geom_dim);
        const size_t dim_size = decmp_coh_._dim_last[_dim]
                              - decmp_coh_._dim_first[_dim] + 1;
        if (4 * rows_mat.size() > 3 * dim_size) {
            decmp_coh_.template compute_full_u_rows<Real>(geom_dim, value_at,
                                           static_cast<size_t>(n_threads_));
            return;
        }

        // Coh-side U is read for decrease_birth; matrix order is
        // reverse filtration order, so the walker truncates from
        // below on non-negate; negate flips.
        if (negate_) {
            decmp_coh_.compute_partial_u_rows(rows_mat, bounds, geom_dim,
                value_at,
                [](Real a, Real b) { return a > b; },
                static_cast<size_t>(n_threads_));
        } else {
            decmp_coh_.compute_partial_u_rows(rows_mat, bounds, geom_dim,
                value_at,
                [](Real a, Real b) { return a < b; },
                static_cast<size_t>(n_threads_));
        }
    }
    // ---------------------------------------------------------------

    bool cmp(Real a, Real b) const
    {
        if (negate_)
            return a > b;
        else
            return a < b;
    }

    ComputeFlags get_flags(const Target& target)
    {
        bool increase_birth = std::accumulate(target.begin(), target.end(), false,
                [this](bool x, auto kv) { return x or kv.second.increase_birth(negate_); });
        bool decrease_birth = std::accumulate(target.begin(), target.end(), false,
                [this](bool x, auto kv) { return x or kv.second.decrease_birth(negate_); });
        bool increase_death = std::accumulate(target.begin(), target.end(), false,
                [this](bool x, auto kv) { return x or kv.second.increase_death(negate_); });

        ComputeFlags result;

        result.compute_cohomology = decrease_birth or increase_birth;
        result.compute_homology_u = increase_death;
        result.compute_cohomology_u = decrease_birth;

        return result;
    }

    ComputeFlags get_flags(const DgmTarget& target)
    {
        bool increase_birth = false;
        bool increase_death = false;
        bool decrease_birth = false;

        for(const auto& [point, target_point]: target) {

            if (cmp(point.birth, target_point.birth)) {
                increase_birth = true;
            } else if (cmp(target_point.birth, point.birth)) {
                decrease_birth = true;
            }

            if (cmp(point.death, target_point.death)) {
                increase_death = true;
            }
        }

        ComputeFlags result;

        result.compute_cohomology = decrease_birth or increase_birth;
        result.compute_homology_u = increase_death;
        result.compute_cohomology_u = decrease_birth;

        return result;
    }

    ComputeFlags get_flags(const Indices& indices, const Values& values)
    {
        if (not decmp_hom_.is_reduced)
            throw std::runtime_error("get_flags(indices, values) requires hom reduced; call ensure_hom_reduced() first");

        bool increase_birth = false;
        bool increase_death = false;
        bool decrease_birth = false;

        for(size_t i = 0 ; i < indices.size() ; ++i) {
            auto simplex_idx = indices[i];
            Real current_value = fil_.get_cell_value(simplex_idx);
            Real target_value = values[i];
            bool is_positive = decmp_hom_.is_positive(simplex_idx);

            if (is_positive and cmp(current_value, target_value)) {
                increase_birth = true;
            } else if (is_positive and cmp(target_value, current_value)) {
                decrease_birth = true;
            } else if (not is_positive and cmp(current_value, target_value)) {
                increase_death = true;
            }
        }

        ComputeFlags result;

        result.compute_cohomology = decrease_birth or increase_birth;
        result.compute_homology_u = increase_death;
        result.compute_cohomology_u = decrease_birth;

        return result;
    }

    dim_type get_dimension(size_t simplex_index) const
    {
        if (fil_.size())
            return fil_.dim_by_sorted_id(simplex_index);
        else
            return 0;
    }

    // Lazily materialize the decompositions and U rows the crit-set
    // walker will read for these (index -> target) moves. Follows the
    // economical recipe: hom is reduced unconditionally (the
    // positive/negative dispatch and the death-side V/U reads need it),
    // coh is reduced only when some move is birth-side, and U rows are
    // solved only for the U-needing directions (increase-death on hom,
    // decrease-birth on coh), grouped by geometric dim because
    // ensure_has_u_* infers a single dim block from its first row.
    // Clearing stays on; ELZ is restored lazily inside ensure_has_u_*.
    void prepare_targets_(const Indices& indices, const Values& values)
    {
        if (not with_crit_sets_)
            throw std::runtime_error(
                "critical sets require with_crit_sets=true at construction "
                "(the optimizer was built dgm-loss only, without V/U)");

        ensure_hom_reduced();

        auto flags = get_flags(indices, values);
        if (flags.compute_cohomology)
            ensure_coh_reduced();

        if (flags.compute_homology_u)
            ensure_u_rows_(indices, values, /*death_side=*/true);
        if (flags.compute_cohomology_u)
            ensure_u_rows_(indices, values, /*death_side=*/false);
    }

    // Select the U-needing moves on one side and solve their U rows,
    // grouped by geometric dim. death_side=true picks increase-death moves
    // (negative simplex moving filtration-forward) -> rows of U_hom.
    // death_side=false picks decrease-birth moves (positive simplex moving
    // filtration-backward) -> rows of U_coh. Positivity is read from the
    // (already reduced) hom pairing.
    void ensure_u_rows_(const Indices& indices, const Values& values, bool death_side)
    {
        std::unordered_map<dim_type, std::pair<Indices, Values>> by_dim;
        for(size_t i = 0 ; i < indices.size() ; ++i) {
            size_t idx = static_cast<size_t>(indices[i]);
            Real target = values[i];
            Real current = fil_.get_cell_value(idx);
            bool selected = death_side
                ? (decmp_hom_.is_negative(idx) and cmp(current, target))
                : (decmp_hom_.is_positive(idx) and cmp(target, current));
            if (not selected)
                continue;
            auto d = fil_.dim_by_sorted_id(indices[i]);
            by_dim[d].first.push_back(indices[i]);
            by_dim[d].second.push_back(target);
        }
        for(auto&& [d, rows_bounds] : by_dim) {
            if (death_side)
                ensure_has_u_hom(d, rows_bounds.first, rows_bounds.second);
            else
                ensure_has_u_coh(d, rows_bounds.first, rows_bounds.second);
        }
    }

    // Dispatch one (index -> value) move. Assumes prepare_targets_ has
    // already reduced the needed side(s) and solved the needed U rows.
    CriticalSet singleton_prepared_(size_t index, Real value)
    {
        if (decmp_hom_.is_negative(index))
            return {value, change_death(index, value)};
        else
            return {value, change_birth(index, value)};
    }

    CriticalSet singleton(size_t index, Real value)
    {
        prepare_targets_({static_cast<Int>(index)}, {value});
        return singleton_prepared_(index, value);
    }

    CriticalSets singletons(const Indices& indices, const Values& values)
    {
        if (indices.size() != values.size())
            throw std::runtime_error("indices and values must have the same size");

        prepare_targets_(indices, values);

        CriticalSets result;
        result.reserve(indices.size());

        for(size_t i = 0 ; i < indices.size() ; ++i) {
            result.emplace_back(singleton_prepared_(static_cast<size_t>(indices[i]), values[i]));
        }

        return result;
    }

    // Invalidate both decompositions and reassign the filtration.
    // The next ensure_*_built / ensure_*_reduced call rebuilds whichever
    // side it needs; no eager reduction here.
    //
    // TODO(revisit): update() is suspected to be broken (sketchy state
    // invariants around boundary_data_ + params_* reset). The lazy-world
    // wiring below preserves the pre-laziness observable behavior for
    // test_diff_update_is_lazy.py; a proper audit is planned separately.
    // Do not extend until that audit lands.
    void update(const Values& new_values, int n_threads = 1)
    {
        (void) n_threads;
        fil_.set_values(new_values);
        boundary_data_ = fil_.boundary_matrix(n_threads_);
        decmp_hom_ = Decomposition();
        decmp_coh_ = Decomposition();
        decmp_hom_built_ = false;
        decmp_coh_built_ = false;
        params_hom_ = Params();
        params_coh_ = Params();
    }

    decltype(auto) convert_critical_sets(const CriticalSets& critical_sets) const
    {
        std::unordered_map<size_t, Values> result;
        for(const auto& crit_set: critical_sets) {
            Real value = crit_set.first;
            for(size_t index: crit_set.second) {
                result[index].push_back(value);
            }
        }
        return result;
    }

    Real get_cell_value(size_t simplex_idx) const
    {
        return fil_.get_cell_value(simplex_idx);
    }

//    Target dgm_target_to_target(const DgmTarget& dgm_target) const
//    {
//        Target target;
//
//        for(auto&& [point, target_point]: dgm_target) {
//            size_t birth_simplex = point.birth;
//            Real current_birth_value = get_cell_value(birth_simplex);
//            Real target_birth_value = target_point.birth;
//
//            if (point.birth != target_point.birth)
//                target.emplace(point.birth_index, {current_birth_value, target_birth_value, true});
//
//            size_t death_simplex = point.death;
//            Real current_death_value = get_cell_value(death_simplex);
//            Real target_death_value = target_point.death;
//
//            if (current_death_value != target_death_value)
//                target.emplace(death_simplex, {current_death_value, target_death_value, false});
//        }
//
//        return target;
//    }

    IndicesValues simplify(Real epsilon, DenoiseStrategy strategy, dim_type dim)
    {
        auto& decmp = ensure_pairing_reduced();

        IndicesValues result;

        auto dgm = decmp.diagram(fil_, false)[dim];

//        causes bugs: spurious points in diagram, need to materialize the diagram
//        for(auto p: decmp_hom_.diagram(fil_, false)[dim]) {
        for(auto p: dgm) {
            if (p.birth_index == p.death_index)
                throw std::runtime_error("bad p in simplify");
            if (p.persistence() <= epsilon) {
                if (strategy == DenoiseStrategy::BirthBirth) {
                    result.push_back(p.death_index, p.birth);
                } else if (strategy == DenoiseStrategy::DeathDeath)
                    result.push_back(p.birth_index, p.death);
                else if (strategy == DenoiseStrategy::Midway) {
                    result.push_back(p.birth_index, (p.birth + p.death) / 2);
                    result.push_back(p.death_index, (p.birth + p.death) / 2);
                }
            }
        }

        return result;
    }

    Real get_nth_persistence(dim_type d, int n)
    {
        auto& decmp = ensure_pairing_reduced();
        return oineus::get_nth_persistence(fil_, decmp, d, n);
    }

    std::pair<IndicesValues, Real> match_and_distance(typename
        Diagrams<Real>::Dgm& template_dgm, dim_type d, Real wasserstein_q,
        Real delta, bool dualize = false)
    {
        // set ids in template diagram
        for(size_t i = 0 ; i < template_dgm.size() ; ++i) {
            template_dgm[i].id = i;

            if (template_dgm[i].is_inf())
                throw std::runtime_error("infinite point in template diagram");
        }

        using Diagram = typename Diagrams<Real>::Dgm;

        IndicesValues result;

        hera::AuctionParams<Real> hera_params;
        hera_params.return_matching = true;
        hera_params.match_inf_points = false;
        hera_params.wasserstein_power = wasserstein_q;
        hera_params.delta = delta;

        // Reduce the requested side if the caller didn't already. The
        // diagram/pairing is identical either way; dualize is the caller's
        // explicit side choice (homology by default), so honor it rather
        // than reusing an unrelated already-reduced side.
        if (dualize)
            ensure_coh_reduced();
        else
            ensure_hom_reduced();
        auto& decmp = dualize ? decmp_coh_ : decmp_hom_;

        Diagram current_dgm = decmp.diagram(fil_, false).get_diagram_in_dimension(d);

        for(size_t i = 0 ; i < current_dgm.size() ; ++i) {
            current_dgm[i].id = i;
        }

        // template_dgm: bidders, a
        // current_dgm: items, b
        Timer timer;
        timer.reset();
        auto hera_res = hera::wasserstein_cost_detailed<Diagram>(template_dgm, current_dgm, hera_params);
        [[maybe_unused]] auto hera_elapsed = timer.elapsed();
        // IC(hera_elapsed);

        for(auto curr_template: hera_res.matching_b_to_a_) {
            auto current_id = curr_template.first;
            auto template_id = curr_template.second;

            if (current_id < 0)
                continue;

            size_t birth_idx = current_dgm.at(current_id).birth_index;
            size_t death_idx = current_dgm.at(current_id).death_index;

            Real birth_target;
            Real death_target;

            if (template_id >= 0) {
                // matched to off-diagonal point of template diagram

                birth_target = template_dgm.at(template_id).birth;
                death_target = template_dgm.at(template_id).death;
            } else {
                // matched to diagonal point of template diagram
                auto curr_proj_id = -template_id - 1;
                Real m = (current_dgm.at(curr_proj_id).birth + current_dgm.at(curr_proj_id).death) / 2;
                birth_target = death_target = m;
            }

            result.push_back(birth_idx, birth_target);
            result.push_back(death_idx, death_target);
        }

        return {result, hera_res.distance};
    }

    IndicesValues match(typename Diagrams<Real>::Dgm& template_dgm, dim_type
        d, Real wasserstein_q, Real delta, bool dualize = false)
    {
        return match_and_distance(template_dgm, d, wasserstein_q, delta, dualize).first;
    }

    IndicesValues combine_loss(const CriticalSets& critical_sets, ConflictStrategy strategy)
    {
        if (strategy != ConflictStrategy::FixCritAvg)
            return combine_loss(critical_sets, Target(), strategy);
        else
            throw std::runtime_error("Need target to use FixCritAvg strategy");
    }

    IndicesValues combine_loss(const CriticalSets& critical_sets, const Target& target, ConflictStrategy strategy)
    {
        CALI_CXX_MARK_FUNCTION;
        auto simplex_to_values = convert_critical_sets(critical_sets);
        IndicesValues indvals;

        if (strategy == ConflictStrategy::Max) {
            for(auto&& [simplex_idx, values]: simplex_to_values) {
                Real current_value = get_cell_value(simplex_idx);
                // compare by displacement from current value
                Real target_value = *std::max_element(values.begin(), values.end(), [current_value](Real a, Real b) { return abs(a - current_value) < abs(b - current_value); });
                indvals.push_back(simplex_idx, target_value);
            }
        } else if (strategy == ConflictStrategy::Avg) {
            for(auto&& [simplex_idx, values]: simplex_to_values) {
                Real target_value = std::accumulate(values.begin(), values.end(), static_cast<Real>(0)) / values.size();
                indvals.emplace_back(simplex_idx, target_value);
            }
        } else if (strategy == ConflictStrategy::Sum) {
            // return all prescribed values, gradient of loss will be summed
            for(auto&& [simplex_idx, values]: simplex_to_values) {
                for(auto value: values) {
                    indvals.emplace_back(simplex_idx, value);
                }
            }
        } else if (strategy == ConflictStrategy::FixCritAvg) {
            // send critical cells according to the matching loss
            // average on others
            for(auto&& [simplex_idx, values]: simplex_to_values) {
                // where matching loss tells critical cells to go
                // is contained in critical_prescribed map
                auto critical_iter = target.find(simplex_idx);
                Real target_value;
                if (critical_iter == target.end())
                    target_value = std::accumulate(values.begin(), values.end(), static_cast<Real>(0)) / values.size();
                else
                    target_value = critical_iter->second.target_value;

                indvals.emplace_back(simplex_idx, target_value);
            }
        }

        return indvals;
    }

    IndicesValues combine_loss(const Indices& indices, const Values& values, ConflictStrategy strategy)
    {
        return combine_loss(singletons(indices, values), strategy);
    }

    // Fused per-pair critical-set walk + conflict resolution, in one
    // C++ call with no intermediate Python lists. Caller responsibility:
    //   - call ensure_reduced_hom(need_u_hom) and ensure_reduced_coh(need_u_coh)
    //     beforehand with flags that match the move directions in
    //     (indices, values), or use ensure_reduced_for_partial_u_*
    //     followed by a partial-U pass when only a subset of rows of
    //     U is needed. This is what the oineus.diff backward does.
    // For FCA the (indices, values) input doubles as the per-critical-
    // simplex target map: each input pair declares one critical simplex
    // whose target is its accompanying value.
    IndicesValues crit_sets_apply(const Indices& indices, const Values& values,
                                  ConflictStrategy strategy)
    {
        CALI_CXX_MARK_FUNCTION;
        if (not with_crit_sets_)
            throw std::runtime_error(
                "crit_sets_apply called on a dgm-loss optimizer "
                "(with_crit_sets_=false); construct the optimizer with "
                "with_crit_sets=true to use the crit-sets backward.");
        if (indices.size() != values.size())
            throw std::runtime_error("crit_sets_apply: indices and values must have the same size");

        // Per-pair dispatch reads decmp_hom_.is_negative(idx), so hom
        // must be at least R-reduced. ensure_hom_reduced is a no-op
        // if the forward already reduced this side.
        ensure_hom_reduced();

        // First pass: per-pair walk, accumulate into a flat (id -> values)
        // multimap. Critical simplices (the input ones, for FCA) are
        // remembered with their prescribed target value.
        std::unordered_map<size_t, Values> per_simplex_targets;
        std::unordered_map<size_t, Real> critical_prescribed;
        per_simplex_targets.reserve(indices.size() * 4);
        if (strategy == ConflictStrategy::FixCritAvg)
            critical_prescribed.reserve(indices.size());

        for (size_t i = 0; i < indices.size(); ++i) {
            size_t idx = indices[i];
            Real target = values[i];
            Real current = fil_.get_cell_value(idx);
            if (current == target)
                continue;

            // cmp(a, b) is true when a is filtration-strictly-less than b
            // (it respects negate). cmp(current, target) thus means
            // "the move is filtration-increasing", i.e. increase_*;
            // cmp(target, current) means decrease_*.
            Indices crit;
            if (decmp_hom_.is_negative(idx)) {
                if (cmp(current, target))
                    crit = increase_death(idx, target);
                else
                    crit = decrease_death(idx, target);
            } else {
                ensure_coh_reduced();
                if (cmp(current, target))
                    crit = increase_birth(idx, target);
                else
                    crit = decrease_birth(idx, target);
            }

            for (auto sid : crit)
                per_simplex_targets[sid].push_back(target);

            if (strategy == ConflictStrategy::FixCritAvg)
                critical_prescribed[idx] = target;
        }

        // Second pass: conflict resolution. Output sizes are bounded by
        // per_simplex_targets.size() except for Sum, which expands to
        // the total number of contributions.
        IndicesValues out;
        if (strategy == ConflictStrategy::Sum) {
            size_t total = 0;
            for (const auto& [sid, vs] : per_simplex_targets)
                total += vs.size();
            out.indices.reserve(total);
            out.values.reserve(total);
        } else {
            out.indices.reserve(per_simplex_targets.size());
            out.values.reserve(per_simplex_targets.size());
        }

        if (strategy == ConflictStrategy::Max) {
            for (auto&& [sid, vs] : per_simplex_targets) {
                Real cur = fil_.get_cell_value(sid);
                Real picked = *std::max_element(vs.begin(), vs.end(),
                    [cur](Real a, Real b) { return std::abs(a - cur) < std::abs(b - cur); });
                out.emplace_back(sid, picked);
            }
        } else if (strategy == ConflictStrategy::Avg) {
            for (auto&& [sid, vs] : per_simplex_targets) {
                Real avg = std::accumulate(vs.begin(), vs.end(), static_cast<Real>(0)) / vs.size();
                out.emplace_back(sid, avg);
            }
        } else if (strategy == ConflictStrategy::Sum) {
            for (auto&& [sid, vs] : per_simplex_targets) {
                for (auto v : vs)
                    out.emplace_back(sid, v);
            }
        } else if (strategy == ConflictStrategy::FixCritAvg) {
            for (auto&& [sid, vs] : per_simplex_targets) {
                auto it = critical_prescribed.find(sid);
                Real picked;
                if (it == critical_prescribed.end())
                    picked = std::accumulate(vs.begin(), vs.end(), static_cast<Real>(0)) / vs.size();
                else
                    picked = it->second;
                out.emplace_back(sid, picked);
            }
        }

        return out;
    }

    Dgms compute_diagram(bool include_inf_points)
    {
        auto& decmp = ensure_pairing_reduced();
        return decmp.diagram(fil_, include_inf_points);
    }

    void reduce_all()
    {
        // reduce_all is a primary work method (not a safety guard); it
        // owns its state machine and materializes both decompositions
        // before reducing them.
        ensure_hom_built();
        ensure_coh_built();
        params_hom_.clearing_opt = false;
        params_hom_.compute_u = params_hom_.compute_v = true;
        if (!decmp_hom_.is_reduced or (params_hom_.compute_u and not decmp_hom_.has_matrix_u())) {
            decmp_hom_.reduce_serial(params_hom_);
        }

        params_coh_.clearing_opt = false;
        params_coh_.compute_u = params_coh_.compute_v = true;
        if (!decmp_coh_.is_reduced or (params_coh_.compute_u and not decmp_coh_.has_matrix_u())) {
            decmp_coh_.reduce_serial(params_coh_);
        }
    }

    // Precondition: decmp_coh_ is reduced. Throws otherwise.
    Indices increase_birth(size_t positive_simplex_idx, Real target_birth) const
    {
        if (not decmp_coh_.is_reduced)
            throw std::runtime_error("increase_birth requires coh reduced; call ensure_coh_reduced() first");
        assert(fil_.cmp(fil_.get_cell_value(positive_simplex_idx), target_birth));

        Indices result;

        // v_col() reads the V column directly from the kept working form (or
        // at-rest v_data), so the fused keep-working decomposition needs no
        // materialization here.
        const auto& vcol = decmp_coh_.v_col(fil_.index_in_matrix(positive_simplex_idx, true));

        for(auto index_in_matrix = vcol.rbegin() ; index_in_matrix != vcol.rend() ; ++index_in_matrix) {
            auto fil_idx = fil_.index_in_filtration(*index_in_matrix, true);
            if (fil_.cmp(target_birth, fil_.get_cell_value(fil_idx)))
                break;

            result.push_back(fil_idx);
        }

        assert(not result.empty());

        return result;
    }

    Indices increase_birth(size_t positive_simplex_idx) const
    {
        return increase_birth(positive_simplex_idx, fil_.infinity());
    }

    // Precondition: decmp_coh_ is reduced. Throws otherwise.
    Indices decrease_birth(size_t positive_simplex_idx, Real target_birth) const
    {
        if (not decmp_coh_.is_reduced)
            throw std::runtime_error("decrease_birth requires coh reduced; call ensure_coh_reduced() first");
        assert(fil_.cmp(target_birth, fil_.get_cell_value(positive_simplex_idx)));

        Indices result;

        for(auto index_in_matrix: decmp_coh_.u_data_t.at(fil_.index_in_matrix(positive_simplex_idx, true))) {
            auto fil_idx = fil_.index_in_filtration(index_in_matrix, true);

            if (fil_.cmp(fil_.get_cell_value(fil_idx), target_birth)) {
                break;
            }

            result.push_back(fil_idx);
        }

        assert(not result.empty());

        return result;
    }

    Indices decrease_birth(size_t positive_simplex_idx) const
    {
        return decrease_birth(positive_simplex_idx, -fil_.infinity());
    }

    // Precondition: decmp_hom_ is reduced. Throws otherwise.
    Indices increase_death(size_t negative_simplex_idx, Real target_death) const
    {
        if (not decmp_hom_.is_reduced)
            throw std::runtime_error("increase_death requires hom reduced; call ensure_hom_reduced() first");
        Indices result;

        const auto& u_rows = decmp_hom_.u_data_t;

        // r_low()/r_is_zero() read R from the kept working form (or at-rest
        // r_data), so no materialization is forced here.
        Int sigma = decmp_hom_.r_low(negative_simplex_idx);

        assert(sigma >= 0 and sigma < static_cast<Int>(decmp_hom_.n_cols_total()));

        for(auto tau_idx: u_rows.at(negative_simplex_idx)) {
            if (fil_.cmp(target_death, fil_.get_cell_value(tau_idx))) {
                break;
            }

            if (decmp_hom_.r_low(tau_idx) <= sigma) {
                result.push_back(tau_idx);
            }
        }

        assert(not result.empty());

        return result;
    }

    Indices increase_death(size_t negative_simplex_idx) const
    {
        CALI_CXX_MARK_FUNCTION;
        return increase_death(negative_simplex_idx, fil_.infinity());
    }

    // Precondition: decmp_hom_ is reduced. Throws otherwise.
    Indices decrease_death(size_t negative_simplex_idx, Real target_death) const
    {
        CALI_CXX_MARK_FUNCTION;
        if (not decmp_hom_.is_reduced)
            throw std::runtime_error("decrease_death requires hom reduced; call ensure_hom_reduced() first");
        Indices result;

        Int sigma = decmp_hom_.r_low(negative_simplex_idx);

        assert(sigma >= 0 and sigma < static_cast<Int>(decmp_hom_.n_cols_total()));

        const auto& vcol = decmp_hom_.v_col(negative_simplex_idx);

        for(auto tau_idx_it = vcol.rbegin() ; tau_idx_it != vcol.rend() ; ++tau_idx_it) {
            auto tau_idx = *tau_idx_it;

            if (fil_.cmp(fil_.get_cell_value(tau_idx), target_death))
                break;

            // explicit check for is_zero is not necessary for signed Int, low returns -1 for empty columns
            if (decmp_hom_.r_low(tau_idx) < sigma or decmp_hom_.r_is_zero(tau_idx))
                continue;

            result.push_back(tau_idx);
        }

        assert(not result.empty());

        return result;
    }

    Indices decrease_death(size_t negative_simplex_idx) const
    {
        CALI_CXX_MARK_FUNCTION;
        return decrease_death(negative_simplex_idx, -fil_.infinity());
    }

    Decomposition get_homology_decompostion() const
    {
        if (not decmp_hom_built_)
            throw std::runtime_error("homology_decomposition unavailable; call ensure_hom_built() or ensure_hom_reduced() first");
        return decmp_hom_;
    }

    Decomposition get_cohomology_decompostion() const
    {
        if (not decmp_coh_built_)
            throw std::runtime_error("cohomology_decomposition unavailable; call ensure_coh_built() or ensure_coh_reduced() first");
        return decmp_coh_;
    }

    bool operator==(const TopologyOptimizer& other) const
    {
        return negate_ == other.negate_
            && fil_ == other.fil_
            && boundary_data_ == other.boundary_data_
            && decmp_hom_ == other.decmp_hom_
            && decmp_coh_ == other.decmp_coh_
            && decmp_hom_built_ == other.decmp_hom_built_
            && decmp_coh_built_ == other.decmp_coh_built_
            && params_hom_ == other.params_hom_
            && params_coh_ == other.params_coh_;
    }

    bool operator!=(const TopologyOptimizer& other) const
    {
        return !(*this == other);
    }

// private:
    // data
    bool negate_;

    Fil fil_;

    // Cached boundary matrix; built eagerly in the ctor from fil_. Both
    // decmp_hom_ and decmp_coh_ are constructed from this on demand; coh
    // antitransposes internally. Shared substrate -> one boundary build
    // per optimizer regardless of which sides end up reduced.
    BoundaryMatrix boundary_data_;

    Decomposition decmp_hom_;
    Decomposition decmp_coh_;

    // True once the corresponding Decomposition has been materialized
    // from boundary_data_. Construction is deferred to the first
    // ensure_*_built / ensure_*_reduced call; safety-guard methods throw
    // when these are false instead of triggering construction.
    bool decmp_hom_built_ {false};
    bool decmp_coh_built_ {false};

    Params params_hom_;
    Params params_coh_;

    // True iff we are set up to drive crit-sets backward (V on the
    // forward side, U recoverable via ensure_has_u_*). False = the
    // optimizer only supports diagram-loss; crit_sets_apply throws.
    bool with_crit_sets_ { true };

    // Thread count used by the reduction drivers. Stored here for the
    // ensure_has_u_* methods, which also feed it into
    // compute_partial_u_rows / compute_full_u_rows.
    int n_threads_ { 1 };

    // Geometric dims (filtration-layout) in which we restore ELZ
    // during the forward reduction, so that partial-U is admissible.
    // For dgm-loss this is unused (restore_elz block in reduce_serial
    // is gated on compute_v).
    DimVec dims_to_restore_elz_;

    // Which equation to solve for U on demand (V^T U^T = I row-form,
    // R U = D column-form, or LegacyInBand which builds U during
    // reduction). The constructor picks the forward recipe accordingly.
    UStrategy u_strategy_ { UStrategy::Auto };

    // methods
    bool cmp(Real a, Real b)
    {
        return negate_ ? a > b : a < b;
    }

    Indices change_birth(size_t positive_simplex_idx, Real target_birth)
    {
        CALI_CXX_MARK_FUNCTION;
        Real current_birth = get_cell_value(positive_simplex_idx);

        if (not decmp_coh_.is_reduced)
            throw std::runtime_error("change_birth requires coh reduced; call ensure_coh_reduced() first");

        if (cmp(target_birth, current_birth))
            return decrease_birth(positive_simplex_idx, target_birth);
        else if (fil_.cmp(current_birth, target_birth))
            return increase_birth(positive_simplex_idx, target_birth);
        else
            return {};
    }

    Indices change_death(size_t negative_simplex_idx, Real target_death)
    {
        CALI_CXX_MARK_FUNCTION;
        Real current_death = get_cell_value(negative_simplex_idx);
        if (cmp(target_death, current_death))
            return decrease_death(negative_simplex_idx, target_death);
        else if (cmp(current_death, target_death))
            return increase_death(negative_simplex_idx, target_death);
        else
            return {};
    }

};

template<class Cell, class Real>
inline std::ostream& operator<<(std::ostream& out, const TopologyOptimizer<Cell, Real>& opt)
{
    out << "TopologyOptimizer(size=" << opt.fil_.size()
        << ", with_crit_sets=" << (opt.with_crit_sets_ ? "true" : "false")
        << ", is_hom_built=" << (opt.is_hom_built() ? "true" : "false")
        << ", is_coh_built=" << (opt.is_coh_built() ? "true" : "false")
        << ", n_threads=" << opt.n_threads_
        << ")";
    return out;
}

} // namespace
