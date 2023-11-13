#pragma once

#include <vector>
#include <unordered_map>
#include <cmath>
#include <numeric>

#include "simplex.h"
#include "sparse_matrix.h"
#include "decomposition.h"
#include "filtration.h"
#include "diagram.h"

// suppress pragma message from boost
#define BOOST_BIND_GLOBAL_PLACEHOLDERS

namespace oineus {

template<typename Cell, typename Real_, int P = 2>
struct KerImCokReduced {
    using Int = typename Cell::Int;
    using Real = Real_;
    using MatrixTraits = SimpleSparseMatrixTraits<Int, P>;
    using Column = typename MatrixTraits::Column;
    using Matrix = typename MatrixTraits::Matrix;
    using Fil = Filtration<Cell, Real>;

    using VRUDecomp = VRUDecomposition<Int>;
    using Point = DgmPoint<Real>;
    using Dgms = oineus::Diagrams<Real>;

public:
    Fil fil_K_;          //Full complex with the function values for F
    Fil fil_L_;          //Sub complex with the function values for G
    VRUDecomp dcmp_F_;   //the reduced triple for F0
    VRUDecomp dcmp_G_;   //reduced triple for G
    VRUDecomp dcmp_im_;  //reduced image triple
    VRUDecomp dcmp_ker_; //reduced kernel triple
    VRUDecomp dcmp_cok_; //reduced cokernel triple
    int max_dim_;        //the maximum dimension of a cell
    Dgms ker_diagrams_;  // kernel diagrams (essentially, map dim -> vector of DgmPoints
    Dgms im_diagrams_;   // image diagrams
    Dgms cok_diagrams_;  // cokernel diagrams
private:
    std::vector<size_t> sorted_K_to_sorted_L_; // given sorted_id i of cell in K, sorted_K_to_sorted_L_[i] is its sorted_id in L; for cells not in K, stores k_invalid_index
    std::vector<size_t> sorted_L_to_sorted_K_; // given sorted_id i of cell in L, sorted_L_to_sorted_K_[i] is its sorted_id in K
    std::vector<size_t> new_order_to_old_;     //given sorted_id i of cell in K, new_order_to_old[i] is its index in the ordering 'first L, then K-L', used in D_im
    std::vector<size_t> old_order_to_new_;     // the inverse of the above
    std::vector<size_t> K_to_ker_column_index_;
    Params params_;
    bool include_zero_persistence_ {false};   // whether we want to have points on the diagonal in the diagram

    // entries (rows) in col are indexed w.r.t. K filtration order
    // return col reindexed by the new order: L first, K-L second
    // TODO: add to MatrixTraits
    Column reindex_to_new_order(const Column& col) const
    {
        Column new_col = col;

        for(auto& x: new_col)
            x = old_order_to_new_[x];

        MatrixTraits::sort(new_col);

        return new_col;
    }

    Matrix compute_d_im() const
    {
        if (not dcmp_F_.is_reduced)
            throw std::runtime_error("reduce D_f first");

        const Matrix& m = dcmp_F_.get_D();

        Matrix result;

        result.reserve(m.size());

        for(const auto& col: m) {
            result.emplace_back(reindex_to_new_order(col));
        }

        return result;
    }

    // take columns from v that correspond to cycles (i.e., the corresponding column in r is zero)
    // apply old_to_new_order to them (reorder rows)
    // return the matrix comprised from the resulting columns
    Matrix compute_d_ker()
    {
        const Matrix& v = dcmp_im_.get_V();
        const Matrix& r = dcmp_im_.get_R();

        Matrix result;

        size_t result_col_idx = 0;

        int n_cycles = 0;
        int n_non_cycles = 0;

        for(size_t col_idx = 0 ; col_idx < v.size() ; ++col_idx) {
            if (not is_zero(r[col_idx])) {
                n_non_cycles++;
                continue;
            }

            n_cycles++;

            result.emplace_back(reindex_to_new_order(v[col_idx]));

            K_to_ker_column_index_[col_idx] = result_col_idx;

            result_col_idx++;
        }

        if (params_.verbose) std::cerr << "compute_d_ker: n_non_cycles = " << n_non_cycles << ", n_cycles = " << n_cycles << std::endl;

        return result;
    }

    Matrix compute_d_cok()
    {
        Matrix d_cok = dcmp_F_.get_D();

        for(size_t i = 0 ; i < d_cok.size() ; i++) {
            auto index_in_L = sorted_K_to_sorted_L_[i];
            // skip non-L columns
            if (index_in_L == k_invalid_index)
                continue;

            // if the column in V for L is not a cycle, skip
            if (not is_zero(dcmp_G_.get_R()[index_in_L]))
                continue;

            // copy column of V_g to a new column
            Column new_col = dcmp_G_.get_V()[index_in_L];

            // indexing in G is with respect to L simplices, must re-index w.r.t. K
            for(auto& x: new_col)
                x = sorted_L_to_sorted_K_[x];

            // NB: sorting not needed: new_col was sorted before

            d_cok[i] = std::move(new_col);
        }
        return d_cok;
    }

public:

    bool is_in_L(size_t index_in_K) const
    {
        return sorted_K_to_sorted_L_[index_in_K] != k_invalid_index;
    }

    bool is_in_K_only(size_t index_in_K) const
    {
        return sorted_K_to_sorted_L_[index_in_K] == k_invalid_index;
    }

    // parameters: complex K, a subcomplex L, reduction params
    KerImCokReduced(const Fil& K, const Fil& L, Params& params, bool include_zero_persistence=false)
            :
            fil_K_(K),
            fil_L_(L),
            sorted_K_to_sorted_L_(K.size(), k_invalid_index),
            sorted_L_to_sorted_K_(L.size(), k_invalid_index),
            new_order_to_old_(K.size(), k_invalid_index),
            old_order_to_new_(K.size(), k_invalid_index),
            K_to_ker_column_index_(K.size(), k_invalid_index),
            max_dim_(K.max_dim()),
            ker_diagrams_(K.max_dim() + 1),
            im_diagrams_(K.max_dim() + 1),
            cok_diagrams_(K.max_dim() + 1),
            params_(params),
            include_zero_persistence_(include_zero_persistence)
    {
        if (params.compute_u) { std::cerr << "WARNING: compute_u will be ignored, do not need it for Ker/Im/Cok algorithm" << std::endl; }
        if (!params.compute_v) { std::cerr << "WARNING: compute_v is false, but V will be computed, we need it for Ker/Im/Cok algorithm" << std::endl; }

        params.compute_v = true;
        params.compute_u = false;

        if (params_.verbose) { std::cerr << "Performing kernel, image, cokernel reduction, reduction parameters: " << params << std::endl; }

        // all cells in L must be present in K
        assert(std::all_of(fil_L_.cells().begin(), fil_L_.cells().end(), [&](const typename Fil::Cell& cell) { return fil_K_.contains_cell_with_uid(cell.get_uid()); }));

        // rough counting check, also in Release mode
        if (fil_K_.size() < fil_L_.size())
            throw std::runtime_error("second argument L must be a subcomplex of the first argument K");

        for(size_t fil_L_idx = 0 ; fil_L_idx < fil_L_.size() ; fil_L_idx++) {//getting sorted L to sorted K is relatively easy
            sorted_L_to_sorted_K_[fil_L_idx] = fil_K_.get_sorted_id_by_uid(fil_L_.get_cell(fil_L_idx).get_uid());
        }

        for(size_t i = 0 ; i < fil_L_.size() ; i++) {
            //for cells in K which are also in L, set the sorted id, which we can get from sorted L to sorted K
            sorted_K_to_sorted_L_[sorted_L_to_sorted_K_[i]] = i;
        }

        if (params_.verbose) { std::cerr << "K_to_L and L_to_K computed" << std::endl; }

        //set up the reduction for F  on K
        dcmp_F_ = VRUDecomp(fil_K_.boundary_matrix_full());
        dcmp_F_.reduce(params);
        if (params_.verbose) { std::cerr << "dcmp_F_ reduced" << std::endl; }

        //set up reduction for G on L
        dcmp_G_ = VRUDecomp(fil_L_.boundary_matrix_full());
        dcmp_G_.reduce(params);
        if (params_.verbose) { std::cerr << "dcmp_G_ reduced" << std::endl; }

        std::iota(new_order_to_old_.begin(), new_order_to_old_.end(), 0);

        if (params_.verbose) std::cerr << "Sorting so that cells in L come before cells in K." << std::endl;

        std::sort(new_order_to_old_.begin(), new_order_to_old_.end(),
                [&](size_t i, size_t j) {
                  if (is_in_L(i) and not is_in_L(j))
                      return true;
                  if (is_in_L(j) and not is_in_L(i))
                      return false;
                  // if both i and j are in L or both are not, use the existing order
                  return i < j;
                });

        // map from old order to new order so that we know which cells correspond to which rows.
        // This could be done by just shuffling the row indices, but as we create a new reduction isntance, we need to create a new matrix anyway.

        for(int i = 0 ; i < fil_K_.size() ; i++) {
            old_order_to_new_[new_order_to_old_[i]] = i;
        }

        params.clearing_opt = false;

        // step 2 of the algorithm
        auto d_im = compute_d_im();
        dcmp_im_ = VRUDecomp(d_im);
        dcmp_im_.reduce(params);
        if (params_.verbose) { std::cerr << "dcmp_im_ reduced" << std::endl; }

        // step 3 of the algorithm

        params.compute_v = false;

        Matrix d_ker = compute_d_ker();
        // NB: d_ker is not a square matrix, has fewer columns that rows. We must give the number of rows (#cells in K) to VRUDecomp ctor.
        dcmp_ker_ = VRUDecomp(d_ker, fil_K_.size());
        dcmp_ker_.reduce(params);
        if (params_.verbose) { std::cerr << "dcmp_ker reduced" << std::endl; }

        // step 4 of the algorithm
        Matrix d_cok = compute_d_cok();
        // NB: d_cok is not a square matrix, has fewer columns that rows. We must give the number of rows to VRUDecomp ctor.
        dcmp_cok_ = VRUDecomp(d_cok, fil_K_.size());
        dcmp_cok_.reduce_parallel_rv(params);
        if (params_.verbose) { std::cerr << "dcmp_cok reduced" << std::endl; }

        if (params.kernel) generate_ker_diagrams();
        if (params.cokernel) generate_cok_diagrams();
        if (params.image) generate_im_diagrams();
    }

    void generate_ker_diagrams(bool inf_points = true)
    {
        if (params_.verbose) std::cerr << "generating kernel diagrams" << std::endl;
        // if we need points at infinity,
        // we have to keep track of the matched positive cells;
        // all unmatched kernel birth cells will give a point at infinity
        std::unordered_set<size_t> matched_positive_cells;

        // simplex tau gives death in Ker(g -> f ) iff τ ∈ L,
        // τ is negative in R_g , and τ is positive in R_f
        for(size_t tau_idx = 0 ; tau_idx < dcmp_G_.get_R().size() ; ++tau_idx) {
            // tau is positive -> skip it
            if (dcmp_G_.is_positive(tau_idx)) {
                if (params_.verbose) std::cerr << "tau_idx not in ker: positive in G" << std::endl;
                continue;
            }

            // always use indices in K in the diagram; tau_idx is w.r.t. L
            size_t death_idx = sorted_L_to_sorted_K_[tau_idx];

            // tau is not positive in R_f -> skip it
            if (dcmp_F_.is_negative(death_idx)) {
                continue;
            }

            // In this case, the lowest one in the column of τ in R_ker
            // corresponds to a simplex σ ∈ K − L that gives birth in Ker(g -> f).
            //Then (σ, τ ) is a pair.

            size_t tau_in_ker_idx = K_to_ker_column_index_[death_idx];
            if (tau_in_ker_idx == k_invalid_index) {
                continue;
            }
            size_t sigma_in_ker_idx = low(dcmp_ker_.get_R()[tau_in_ker_idx]);
            size_t birth_idx = new_order_to_old_[sigma_in_ker_idx];

            Real birth = fil_K_.value_by_sorted_id(birth_idx);
            Real death = fil_K_.value_by_sorted_id(death_idx);

            dim_type dim = fil_K_.get_cell(death_idx).dim() - 1;

            if (birth != death or include_zero_persistence_)
                ker_diagrams_.add_point(dim, birth, death, birth_idx, death_idx);

            if (inf_points) {
                assert(matched_positive_cells.count(birth_idx) == 0);
                matched_positive_cells.insert(birth_idx);
            }

            // for kernel p-diagram, birth and death simplices are
            // (p+1)-simplices
            assert(fil_K_.get_cell(death_idx).dim() == fil_K_.get_cell(birth_idx).dim());
        }

        if (params_.verbose) std::cerr << "finite points in kernel diagram generated, found " << matched_positive_cells.size() << std::endl;

        if (inf_points) {
            for(size_t birth_idx = 0 ; birth_idx < dcmp_F_.get_R().size() ; ++birth_idx) {
                // sigma is in L, skip it
                if (is_in_L(birth_idx))
                    continue;

                // sigma is paired
                if (matched_positive_cells.count(birth_idx))
                    continue;

                // sigma is positive in R_f, skip it
                if (dcmp_F_.is_positive(birth_idx))
                    continue;

                size_t low_idx = low(dcmp_im_.get_R()[birth_idx]);

                // lowest one if R_im is in K, not in L
                // order of rows in R_im: L first, then K-L
                if (low_idx >= fil_L_.size())
                    continue;

                Real birth = fil_K_.get_cell_value(birth_idx);

                dim_type dim = fil_K_.get_cell(birth_idx).dim() - 1;
                ker_diagrams_.add_point(dim, birth, fil_K_.infinity(), birth_idx, k_invalid_index);
            }
        }

        if (params_.verbose) {
            std::cerr << "The kernel diagrams are: " << std::endl;
            for(int i = 0 ; i <= max_dim_ ; i++) {
                std::cerr << "Diagram in dimension " << i << " is: [" << std::endl;
                for(int j = 0 ; j < ker_diagrams_[i].size() ; j++) {
                    std::cerr << ker_diagrams_[i][j] << std::endl;
                }
                std::cerr << "]";
            }
        }
    }

    void generate_cok_diagrams(bool inf_points = true)
    {
        if (params_.verbose) std::cerr << "generating cokernel diagrams" << std::endl;
        // if we need points at infinity,
        // we have to keep track of the matched positive cells;
        // all unmatched kernel birth cells will give a point at infinity
        std::unordered_set<size_t> matched_positive_cells;

        // simplex τ gives death in Cok(g -> f) iff τ is
        // negative in R_f and the lowest one in its column in R_im
        // corresponds to a simplex in K − L
        for(size_t death_idx = 0 ; death_idx < dcmp_F_.get_R().size() ; ++death_idx) {
            // tau is positive -> skip it
            if (dcmp_F_.is_positive(death_idx))
                continue;

            const auto& tau_col_R_im = dcmp_im_.get_R()[death_idx];


            // TODO: get rid of this if
            // tau is positive in R_f -> skip it
            if (dcmp_im_.is_positive(death_idx)) {
                throw std::runtime_error("probably should not happen: negative in R_f implies negative in R_im");
                continue;
            }

            auto im_low = low(tau_col_R_im);

            // lowest one in the column of tau in R_im is in L, skip it
            if (im_low < fil_L_.size())
                continue;

            // In this case, the lowest one in the column of τ in R_cok corresponds to a
            // simplex σ that gives birth in Cok(g -> f). Then (σ, τ) is a pair.
            // row and column order in R_cok, D_cok is the same as in K
            if (is_zero(dcmp_cok_.get_R()[death_idx]))
                continue;
            auto birth_idx = low(dcmp_cok_.get_R()[death_idx]);

            Real birth = fil_K_.value_by_sorted_id(birth_idx);
            Real death = fil_K_.value_by_sorted_id(death_idx);

            dim_type dim = fil_K_.get_cell(birth_idx).dim();

            if (birth != death or include_zero_persistence_)
                cok_diagrams_.add_point(dim, birth, death, birth_idx, death_idx);

            if (inf_points) {
                assert(matched_positive_cells.count(birth_idx) == 0);
                matched_positive_cells.insert(birth_idx);
            }

            // for cokernel p-diagram, birth is at p-simplex, death is at (p+1)-simplex
            assert(fil_K_.get_cell(death_idx).dim() == dim + 1);
        }

        if (params_.verbose) std::cerr << "cokernel diagrams, finite points done, # matched " << matched_positive_cells.size() << std::endl;

        if (inf_points) {
            // A simplex σ gives birth in Cok(g -> f) iff σ is positive
            // in R_f and it is either in K − L or negative in R_g .
            for(size_t birth_idx = 0 ; birth_idx < dcmp_F_.get_R().size() ; ++birth_idx) {
                // sigma is paired, skip it
                if (matched_positive_cells.count(birth_idx))
                    continue;

                // sigma is negative in R_f, skip it
                if (dcmp_F_.is_negative(birth_idx))
                    continue;

                if (not is_in_K_only(birth_idx)) {
                    // sigma is in K and in L, and sigma is positive in R_g, skip it
                    if (dcmp_G_.is_positive(sorted_K_to_sorted_L_[birth_idx]))
                        continue;
                }

                Real birth = fil_K_.value_by_sorted_id(birth_idx);
                dim_type dim = fil_K_.get_cell(birth_idx).dim();
                // K.infinity() will return +inf or -inf depending on
                // negate; plus_inf is max of size_t
                cok_diagrams_.add_point(dim, birth, fil_K_.infinity(), birth_idx, plus_inf);
            }
        }
    }

    // TODO: merge with cokernel diagrams? the logic is close, instead of continue we can just have an if-statement
    void generate_im_diagrams(bool inf_points = true)
    {
        std::unordered_set<size_t> matched_positive_cells;

        // Death. A simplex τ gives death in Im(g →f ) iff τ is negative in Rf
        // and the lowest one in its column in R_im corresponds to a simplex σ ∈ L.
        // Then (σ, τ ) is a pair.
        for(size_t death_idx = 0 ; death_idx < dcmp_F_.get_R().size() ; ++death_idx) {
            // tau is positive in R_f -> skip it
            if (dcmp_F_.is_positive(death_idx))
                continue;

            // tau is positive in R_f -> skip it
            if (dcmp_im_.is_positive(death_idx)) {
                throw std::runtime_error("should not happen: negative in R_f implies negative in R_im");
            }

            auto im_low = low(dcmp_im_.get_R()[death_idx]);

            // lowest one in the column of tau in R_im is not in L, skip it
            if (im_low >= fil_L_.size())
                continue;

            auto birth_idx = new_order_to_old_[im_low];

            if (birth_idx == k_invalid_index)
                throw std::runtime_error("indexing error in generate_im_diagrams");

            Real birth = fil_K_.value_by_sorted_id(birth_idx);
            Real death = fil_K_.value_by_sorted_id(death_idx);
            dim_type dim = fil_K_.get_cell(birth_idx).dim();

            if (birth != death or include_zero_persistence_)
                im_diagrams_.add_point(dim, birth, death, birth_idx, death_idx);

            if (inf_points) {
                assert(matched_positive_cells.count(birth_idx) == 0);
                matched_positive_cells.insert(birth_idx);
            }

            // for image p-diagram, birth is at p-simplex, death is at (p+1)-simplex
            assert(fil_K_.get_cell(death_idx).dim() == dim + 1);
        }

        if (params_.verbose) std::cerr << "image diagrams, finite points done, # matched " << matched_positive_cells.size() << std::endl;

        if (inf_points) {
            int n_inf_points = 0;
            //Birth. A simplex σ gives birth in Im(g →f ) iff σ ∈ L and σ is positive in Rg .
            for(size_t sigma_L_idx = 0 ; sigma_L_idx < dcmp_G_.get_R().size() ; ++sigma_L_idx) {
                size_t birth_idx = sorted_L_to_sorted_K_[sigma_L_idx];
                // sigma is paired, skip it
                if (matched_positive_cells.count(birth_idx))
                    continue;

                // sigma is negative in R_g, skip it
                if (dcmp_G_.is_negative(sigma_L_idx))
                    continue;

                Real birth = fil_K_.value_by_sorted_id(birth_idx);
                dim_type dim = fil_K_.get_cell(birth_idx).dim();
                // K.infinity() will return +inf or -inf depending on
                // negate; plus_inf is max of size_t
                im_diagrams_.add_point(dim, birth, fil_K_.infinity(), birth_idx, plus_inf);
                n_inf_points++;
            }
            if (params_.verbose) std::cerr << "image diagrams, infinite points done, # points at infinity " << n_inf_points << std::endl;
        }
    } // image diagrams

    const Dgms& get_kernel_diagrams() const
    {
        if (not params_.kernel)
            throw std::runtime_error("kernel diagrams were not computed because params.kernel was false in constructor");
        return ker_diagrams_;
    }

    const Dgms& get_image_diagrams() const
    {
        if (not params_.image)
            throw std::runtime_error("image diagrams were not computed because params.image was false in constructor");
        return im_diagrams_;
    }

    const Dgms& get_cokernel_diagrams() const
    {
        if (not params_.cokernel)
            throw std::runtime_error("cokernel diagrams were not computed because params.cokernel was false in constructor");
        return cok_diagrams_;
    }
};
} // namespace oineus
