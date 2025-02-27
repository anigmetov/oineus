#pragma once

#include <vector>
#include <unordered_map>
#include <cmath>
#include <numeric>
#include <ostream>

#include "taskflow/taskflow.hpp"
#include "taskflow/algorithm/for_each.hpp"

#include "simplex.h"
#include "sparse_matrix.h"
#include "decomposition.h"
#include "filtration.h"
#include "diagram.h"
#include "profile.h"

// suppress pragma message from boost
#define BOOST_BIND_GLOBAL_PLACEHOLDERS

namespace oineus {

struct KICRParams {
    bool codomain {false}; // compute dcmp_F (diagrams of the ambient filtration) --- redundant for ker-im-cok
    bool kernel {true};
    bool image  {true};
    bool cokernel {true};
    bool include_zero_persistence {false};
    bool verbose {false};
    bool sanity_check {false};
    int n_threads {1};
    Params params_f;
    Params params_g;
    Params params_ker;
    Params params_im;
    Params params_cok;
};

inline std::ostream& operator<<(std::ostream& out, const KICRParams& p)
{
    out << "KICRParams(compute_kernel = " << p.kernel;
    out << ", compute_image = " << p.image;
    out << ", compute_cokernel = " << p.cokernel;
    out << ", include_zero_persistence = " << p.include_zero_persistence;
    out << ", verbose = " << p.verbose;
    out << ", n_threads = " << p.n_threads;
    out << ", params_f = " << p.params_f;
    out << ", params_g = " << p.params_g;
    if (p.kernel) out << ", params_ker = " << p.params_ker;
    if (p.image)  out << ", params_im = " << p.params_im;
    if (p.cokernel)  out << ", params_cok = " << p.params_cok;
    out << ")";
    return out;
}

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
    typename VRUDecomp::MatrixData dcmp_F_D_;   //the reduced triple for F0
    VRUDecomp dcmp_G_;   //reduced triple for G
    VRUDecomp dcmp_im_;  //reduced image triple
    VRUDecomp dcmp_ker_; //reduced kernel triple
    VRUDecomp dcmp_cok_; //reduced cokernel triple
    int max_dim_;        //the maximum dimension of a cell
    Dgms dom_diagrams_;  // diagrams of the included complex L
    Dgms cod_diagrams_;  // diagrams of the ambient complex K
    Dgms ker_diagrams_;  // kernel diagrams (essentially, map dim -> vector of DgmPoints)
    Dgms im_diagrams_;   // image diagrams
    Dgms cok_diagrams_;  // cokernel diagrams
private:
    std::vector<size_t> sorted_K_to_sorted_L_; // given sorted_id i of cell in K, sorted_K_to_sorted_L_[i] is its sorted_id in L; for cells not in K, stores k_invalid_index
    std::vector<size_t> sorted_L_to_sorted_K_; // given sorted_id i of cell in L, sorted_L_to_sorted_K_[i] is its sorted_id in K
    std::vector<size_t> new_order_to_old_;     //given sorted_id i of cell in K, new_order_to_old[i] is its index in the ordering 'first L, then K-L', used in D_im
    std::vector<size_t> old_order_to_new_;     // the inverse of the above
    std::vector<size_t> K_to_ker_column_index_;
    KICRParams params_;

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
        CALI_CXX_MARK_FUNCTION;
        const Matrix& m = dcmp_F_D_;

        Matrix result;

        result.reserve(m.size());

        for(const auto& col: m) {
            result.emplace_back(reindex_to_new_order(col));
        }

        return result;
    }

    // take columns from v that correspond to cycles (i.e., the corresponding column in r is zero)
    // apply old_to_new_order to them (reorder rows)
    // return the matrix comprised of the resulting columns
    Matrix compute_d_ker()
    {
        CALI_CXX_MARK_FUNCTION;
        const Matrix& v = dcmp_im_.get_V();
        const Matrix& r = dcmp_im_.get_R();

        Matrix result;

        size_t result_col_idx {0}, n_cycles {0}, n_non_cycles {0};

        if (params_.params_ker.n_threads > 1) {
            // can do with prefix sum, but this should be fast enough to keep serial
            for(size_t col_idx = 0 ; col_idx < v.size() ; ++col_idx) {
                if (is_zero(r[col_idx])) {
                    n_cycles++;
                    K_to_ker_column_index_[col_idx] = result_col_idx;
                    result_col_idx++;
                }
            }

            n_non_cycles = v.size() - n_cycles;

            tf::Executor executor(params_.params_ker.n_threads);
            tf::Taskflow taskflow_d_ker;

            result = Matrix(n_cycles, Column());

            taskflow_d_ker.for_each_index((size_t)0, v.size(), (size_t)1,
                    [this, &v, &result](size_t col_idx) {
                        size_t result_idx = K_to_ker_column_index_[col_idx];
                        if (result_idx != k_invalid_index)
                            result[K_to_ker_column_index_[col_idx]] = reindex_to_new_order(v[col_idx]);
                    });

            executor.run(taskflow_d_ker).get();
        } else {
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
        }
        if (params_.verbose) std::cerr << "compute_d_ker: n_non_cycles = " << n_non_cycles << ", n_cycles = " << n_cycles << std::endl;
        return result;
    }

    Matrix compute_d_cok()
    {
        CALI_CXX_MARK_FUNCTION;
        Matrix d_cok = dcmp_F_D_;

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

            // sorting not needed: new_col was sorted before
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
    KerImCokReduced(const Fil& K, const Fil& L,
                    KICRParams& params)
            :
            fil_K_(K),
            fil_L_(L),
            max_dim_(K.max_dim()),
            ker_diagrams_(K.max_dim() + 1),
            im_diagrams_(K.max_dim() + 1),
            cok_diagrams_(K.max_dim() + 1),
            sorted_K_to_sorted_L_(K.size(), k_invalid_index),
            sorted_L_to_sorted_K_(L.size(), k_invalid_index),
            new_order_to_old_(K.size(), k_invalid_index),
            old_order_to_new_(K.size(), k_invalid_index),
            K_to_ker_column_index_(K.size(), k_invalid_index),
            params_(params)
    {
        if (params_.verbose) { std::cerr << "Performing kernel, image, cokernel reduction, reduction parameters: " << params << std::endl; }

        // all cells in L must be present in K
        if (params_.sanity_check) {
            if (!std::all_of(fil_L_.cells().begin(), fil_L_.cells().end(), [&](const typename Fil::Cell& cell) { return fil_K_.contains_cell_with_uid(cell.get_uid()); }))
                throw std::runtime_error("Ker-Im-Cok: L is not a subcomplex of K");
        } else
            assert(std::all_of(fil_L_.cells().begin(), fil_L_.cells().end(), [&](const typename Fil::Cell& cell) { return fil_K_.contains_cell_with_uid(cell.get_uid()); }));

        // rough counting check, also in Release mode
        if (fil_K_.size() < fil_L_.size())
            throw std::runtime_error("second argument L must be a subcomplex of the first argument K");

        if (params_.n_threads > 1) {
            params_.params_f.n_threads = params_.params_g.n_threads = params_.n_threads;
            params_.params_ker.n_threads = params_.params_cok.n_threads = params_.params_im.n_threads = params_.n_threads;
        }

        CALI_MARK_BEGIN("sorted_L_to_sorted_K");
        for(size_t fil_L_idx = 0 ; fil_L_idx < fil_L_.size() ; fil_L_idx++) {//getting sorted L to sorted K is relatively easy
            sorted_L_to_sorted_K_[fil_L_idx] = fil_K_.get_sorted_id_by_uid(fil_L_.get_cell(fil_L_idx).get_uid());
        }
        CALI_MARK_END("sorted_L_to_sorted_K");

        CALI_MARK_BEGIN("sorted_K_to_sorted_L");
        for(size_t i = 0 ; i < fil_L_.size() ; i++) {
            //for cells in K which are also in L, set the sorted id, which we can get from sorted L to sorted K
            sorted_K_to_sorted_L_[sorted_L_to_sorted_K_[i]] = i;
        }
        CALI_MARK_END("sorted_K_to_sorted_L");

        if (params_.verbose) { std::cerr << "K_to_L and L_to_K computed" << std::endl; }

        CALI_MARK_BEGIN("fil_K_.boundary_matrix");
        dcmp_F_D_ = fil_K_.boundary_matrix(params_.n_threads);
        CALI_MARK_END("fil_K_.boundary_matrix");

        //set up the reduction for F  on K
        if (params.codomain) {
            CALI_MARK_BEGIN("dcmp_F.reduce");
            dcmp_F_ = VRUDecomp(dcmp_F_D_);
            dcmp_F_.reduce(params_.params_f);
            if (params_.verbose) { std::cerr << "dcmp_F_ reduced with params = " << params_.params_f << std::endl; }
            CALI_MARK_END("dcmp_F.reduce");
        }

        if (params_.verbose) std::cerr << "starting dcmpG" << std::endl;

        //set up reduction for G on L
        CALI_MARK_BEGIN("dcmp_G.reduce");
        params_.params_g.compute_v = params_.params_g.compute_v or params_.cokernel;
        dcmp_G_ = VRUDecomp(fil_L_.boundary_matrix(params_.n_threads));
        dcmp_G_.reduce(params_.params_g);
        if (params_.verbose) { std::cerr << "dcmp_G_ reduced with params = " << params_.params_g << std::endl; }
        CALI_MARK_END("dcmp_G.reduce");

        if (params.codomain) {
            if (params_.verbose) std::cerr << "starting dcmp_F diagram" << std::endl;
            CALI_MARK_BEGIN("dcmp_F.diagram");
            cod_diagrams_ = dcmp_F_.diagram(fil_K_, true);
            if (params_.verbose) { std::cerr << "cod_diagrams computed" << std::endl; }
            CALI_MARK_END("dcmp_F.diagram");
        }

        CALI_MARK_BEGIN("dcmp_G.diagram");
        dom_diagrams_ = dcmp_G_.diagram(fil_L_, true);
        if (params_.verbose) { std::cerr << "dom_diagrams computed" << std::endl; }
        CALI_MARK_END("dcmp_G.diagram");

        CALI_MARK_BEGIN("new_order_to_old");
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
        CALI_MARK_END("new_order_to_old");

        // map from old order to new order so that we know which cells correspond to which rows.
        // This could be done by just shuffling the row indices, but as we create a new reduction isntance, we need to create a new matrix anyway.

        for(size_t i = 0 ; i < fil_K_.size(); i++) {
            old_order_to_new_[new_order_to_old_[i]] = i;
        }

        if (params_.verbose) std::cerr << "starting dcmp_im" << std::endl;
        // step 2 of the algorithm
        CALI_MARK_BEGIN("dcmp_im");
        // TODO: add clearing here; requires more refined L-before-K order (partitioned by dimension)
        params_.params_im.clearing_opt = false;
        // if user wants to compute v, keep it, but if we need ker, we must compute it anyway
        params_.params_im.compute_v = params_.params_im.compute_v or params_.kernel;
        auto d_im = compute_d_im();
        dcmp_im_ = VRUDecomp(d_im);
        dcmp_im_.reduce(params_.params_im);
        if (params_.verbose) { std::cerr << "dcmp_im_ reduced, size=" << dcmp_im_.size() << ", params = " << params_.params_im << std::endl; }
        CALI_MARK_END("dcmp_im");

        // step 3 of the algorithm

        if (params_.verbose) std::cerr << "starting dcmp_ker" << std::endl;
        if (params_.kernel) {
            CALI_MARK_BEGIN("dcmp_ker");
            params_.params_ker.clearing_opt = false;
            Matrix d_ker = compute_d_ker();
            // NB: d_ker is not a square matrix, has fewer columns that rows. We must give the number of rows (#cells in K) to VRUDecomp ctor.
            dcmp_ker_ = VRUDecomp(d_ker, fil_K_.size());
            dcmp_ker_.reduce(params_.params_ker);
            if (params_.verbose) { std::cerr << "dcmp_ker reduced, size=" << dcmp_ker_.size() << ", params = " << params_.params_ker << std::endl; }
            CALI_MARK_END("dcmp_ker");
        }

        if (params_.cokernel) {
            // step 4 of the algorithm
            if (params_.verbose) std::cerr << "starting dcmp_cok" << std::endl;
            CALI_MARK_BEGIN("dcmp_cok");
            params_.params_cok.clearing_opt = false;
            Matrix d_cok = compute_d_cok();
            // NB: d_cok is not a square matrix, has fewer columns that rows. We must give the number of rows to VRUDecomp ctor.
            dcmp_cok_ = VRUDecomp(d_cok, fil_K_.size());
            dcmp_cok_.reduce(params_.params_cok);
            if (params_.verbose) { std::cerr << "dcmp_cok reduced, size=" << dcmp_cok_.size() << std::endl; }
            CALI_MARK_END("dcmp_cok");
        }

        if (params_.verbose and params_.sanity_check) {
            std::cerr << "starting sanity check..." << std::endl;
            sanity_check();
            std::cerr << "sanity check OK" << std::endl;
        }


        if (params_.verbose) std::cerr << "starting kic diagrams" << std::endl;

        if (params_.kernel) generate_ker_diagrams();
        if (params_.cokernel) generate_cok_diagrams();
        if (params_.image) generate_im_diagrams();
    }

    void sanity_check()
    {
        const auto& R_g = dcmp_G_.get_R();
        const auto& R_f = dcmp_F_.get_R();
        const auto& R_im = dcmp_im_.get_R();
        const auto& R_ker = dcmp_ker_.get_R();

        if (params_.codomain and params_.kernel) {
            for(size_t r_g_idx = 0; r_g_idx < R_g.size(); ++r_g_idx) {
                if (is_zero(R_g.at(r_g_idx))) {
                    auto r_f_idx = sorted_L_to_sorted_K_.at(r_g_idx);
                    if (not is_zero(R_f.at(r_f_idx))) {
                        std::cerr << "r_f_idx = " << r_f_idx << ", r_g_idx = " << r_g_idx << std::endl;
                        throw std::runtime_error("condition R_g.at(i) = 0 -> R_f.at(i) =  0 violated");
                    }
                }
            }
        }
        std::cerr << "condition (i) ok" << std::endl;

        if (params_.codomain) {
            for(size_t r_f_idx = 0; r_f_idx < R_f.size(); ++r_f_idx) {
                if (is_zero(R_f.at(r_f_idx)) and not is_zero(R_im.at(r_f_idx))) {
                    std::cerr << "r_f_idx = " << r_f_idx << std::endl;
                    throw std::runtime_error("condition R_f.at(i) = 0 <-> R_f.at(i) =  0 violated: zero in R_f, not zero in R_im");
                }
                if (not is_zero(R_f.at(r_f_idx)) and is_zero(R_im.at(r_f_idx))) {
                    std::cerr << "r_f_idx = " << r_f_idx << std::endl;
                    throw std::runtime_error("condition R_f.at(i) = 0 <-> R_f.at(i) =  0 violated: not zero in R_f, zero in R_im");
                }
            }
        }
        if (params_.verbose) std::cerr << "condition (ii) ok" << std::endl;

        if (params_.kernel) {
            for(size_t r_ker_idx = 0; r_ker_idx < R_ker.size(); ++r_ker_idx) {
                if (is_zero(R_ker.at(r_ker_idx))) {
                    std::cerr << r_ker_idx << std::endl;
                    throw std::runtime_error("zero column in R_ker");
                }
            }
        }

        if (params_.verbose) std::cerr << "condition (iv) ok" << std::endl;

        if (params_.codomain) {
            for(size_t sigma_L_idx = 0; sigma_L_idx < fil_L_.size(); ++sigma_L_idx) {
                size_t index_in_K = sorted_L_to_sorted_K_[sigma_L_idx];

                if (not is_zero(R_f[index_in_K])) {
                    size_t low_in_im = low(R_im[index_in_K]);
                    if (low_in_im >= fil_L_.size()) {
//                        IC(sigma_L_idx, index_in_K, low_in_im);
                        throw std::runtime_error("condition (iii) violated, lowest one in R_im[i] not in L");
                    }
                }

                if (not is_zero(R_g[sigma_L_idx]) and is_zero(R_f[index_in_K])) {
                    size_t low_ker = low(R_ker[K_to_ker_column_index_.at(index_in_K)]);
                    if (low_ker < fil_L_.size()) {
//                        IC(sigma_L_idx, index_in_K, low_ker, fil_L_.size(), fil_K_.size());
                        throw std::runtime_error("condition (vi) violated, lowest one in R_ker[i] is not in K-L");
                    }
                }
            }
        }
        if (params_.verbose) std::cerr << "conditions (iii), (vi) ok" << std::endl;

        if (params_.kernel) {
            for(size_t sigma_K_idx = 0; sigma_K_idx < fil_K_.size(); ++sigma_K_idx) {
                if (is_in_L(sigma_K_idx))
                    continue;

                size_t r_ker_idx = K_to_ker_column_index_.at(sigma_K_idx);
                if (r_ker_idx == k_invalid_index)
                    continue;

                size_t low_ker = low(R_ker.at(r_ker_idx));
                if (low_ker != old_order_to_new_.at(sigma_K_idx)) {
//                    IC(sigma_K_idx, old_order_to_new_[sigma_K_idx], low_ker);
                    throw std::runtime_error("condition (v) violated, lowest one in R_ker[i] is not sigma_i");
                }
            }
        }
        std::cerr << "condition (v) ok" << std::endl;
    }

    void generate_ker_diagrams(bool inf_points = true)
    {
        CALI_CXX_MARK_FUNCTION;
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
//                if (params_.verbose) std::cerr << "tau_idx not in ker: positive in G" << std::endl;
                continue;
            }

            // always use indices in K in the diagram; tau_idx is w.r.t. L
            size_t death_idx = sorted_L_to_sorted_K_[tau_idx];

            // tau is not positive in R_f -> skip it
            if (dcmp_im_.is_negative(death_idx)) {
                continue;
            }

            // In this case, the lowest one in the column of τ in R_ker
            // corresponds to a simplex σ ∈ K − L that gives birth in Ker(g -> f).
            //Then (σ, τ ) is a pair.

            size_t tau_in_ker_idx = K_to_ker_column_index_.at(death_idx);
            if (tau_in_ker_idx == k_invalid_index) {
                continue;
            }
            size_t sigma_in_ker_idx = low(dcmp_ker_.get_R()[tau_in_ker_idx]);
            size_t birth_idx = new_order_to_old_.at(sigma_in_ker_idx);

            Real birth = fil_K_.value_by_sorted_id(birth_idx);
            Real death = fil_K_.value_by_sorted_id(death_idx);

            dim_type dim = fil_K_.get_cell(death_idx).dim() - 1;

            if (birth != death or params_.include_zero_persistence)
                ker_diagrams_.add_point(dim, birth, death, birth_idx, death_idx, fil_K_.get_id_by_sorted_id(birth_idx), fil_K_.get_id_by_sorted_id(death_idx));

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
            for(size_t birth_idx = 0 ; birth_idx < dcmp_im_.get_R().size() ; ++birth_idx) {
                // sigma is in L, skip it
                if (is_in_L(birth_idx))
                    continue;

                // sigma is paired
                if (matched_positive_cells.count(birth_idx))
                    continue;

                // sigma is positive in R_f, skip it
                if (dcmp_im_.is_positive(birth_idx))
                    continue;

                size_t low_idx = low(dcmp_im_.get_R()[birth_idx]);

                // lowest one if R_im is in K, not in L
                // order of rows in R_im: L first, then K-L
                if (low_idx >= fil_L_.size())
                    continue;

                Real birth = fil_K_.get_cell_value(birth_idx);

                dim_type dim = fil_K_.get_cell(birth_idx).dim() - 1;
                ker_diagrams_.add_point(dim, birth, fil_K_.infinity(), birth_idx, k_invalid_index, fil_K_.get_id_by_sorted_id(birth_idx), k_invalid_index);
            }
        }

//        if (params_.verbose) {
//            std::cerr << "The kernel diagrams are: " << std::endl;
//            for(int i = 0 ; i <= max_dim_ ; i++) {
//                std::cerr << "Diagram in dimension " << i << " is: [" << std::endl;
//                for(int j = 0 ; j < ker_diagrams_[i].size() ; j++) {
//                    std::cerr << ker_diagrams_[i][j] << std::endl;
//                }
//                std::cerr << "]";
//            }
//        }
    }

    void generate_cok_diagrams(bool inf_points = true)
    {
        CALI_CXX_MARK_FUNCTION;
        if (params_.verbose) std::cerr << "generating cokernel diagrams" << std::endl;
        // if we need points at infinity,
        // we have to keep track of the matched positive cells;
        // all unmatched kernel birth cells will give a point at infinity
        std::unordered_set<size_t> matched_positive_cells;

        const auto& R_im = dcmp_im_.get_R();
        const auto& R_cok = dcmp_cok_.get_R();

        // simplex τ gives death in Cok(g -> f) iff τ is
        // negative in R_f and the lowest one in its column in R_im
        // corresponds to a simplex in K − L
        for(size_t death_idx = 0 ; death_idx < R_im.size() ; ++death_idx) {
            // tau is positive -> skip it
            if (dcmp_im_.is_positive(death_idx)) {
                continue;
            }

            auto im_low = low(R_im.at(death_idx));

            // lowest one in the column of tau in R_im is in L, skip it
            if (im_low < static_cast<Int>(fil_L_.size())) {
                continue;
            }

            // In this case, the lowest one in the column of τ in R_cok corresponds to a
            // simplex σ that gives birth in Cok(g -> f). Then (σ, τ) is a pair.
            // row and column order in R_cok, D_cok is the same as in K
            if (is_zero(R_cok.at(death_idx)))
                continue;
            auto birth_idx = low(R_cok[death_idx]);

            Real birth = fil_K_.value_by_sorted_id(birth_idx);
            Real death = fil_K_.value_by_sorted_id(death_idx);

            dim_type dim = fil_K_.get_cell(birth_idx).dim();

            if (birth != death or params_.include_zero_persistence)
                cok_diagrams_.add_point(dim, birth, death, birth_idx, death_idx, fil_K_.get_id_by_sorted_id(birth_idx), fil_K_.get_id_by_sorted_id(death_idx));

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
            // in R_f (eqv.: R_im) and it is either in K − L or negative in R_g .
            for(size_t birth_idx = 0 ; birth_idx < R_im.size() ; ++birth_idx) {
                // sigma is paired, skip it
                if (matched_positive_cells.count(birth_idx)) {
                    continue;
                }

                // sigma is negative in R_im, skip it
                if (dcmp_im_.is_negative(birth_idx)) {
                    continue;
                }

                if (not is_in_K_only(birth_idx)) {
                    // sigma is in K and in L, and sigma is positive in R_g, skip it
                    if (dcmp_G_.is_positive(sorted_K_to_sorted_L_.at(birth_idx)))
                        continue;
                }

                Real birth = fil_K_.value_by_sorted_id(birth_idx);
                dim_type dim = fil_K_.get_cell(birth_idx).dim();
                // K.infinity() will return +inf or -inf depending on
                // negate; plus_inf is max of size_t
                cok_diagrams_.add_point(dim, birth, fil_K_.infinity(), birth_idx, plus_inf, fil_K_.get_id_by_sorted_id(birth_idx), plus_inf);
            }
        }
    }

    // TODO: merge with cokernel diagrams? the logic is close, instead of continue we can just have an if-statement
    void generate_im_diagrams(bool inf_points = true)
    {
        CALI_CXX_MARK_FUNCTION;
        std::unordered_set<size_t> matched_positive_cells;

        const auto& R_im = dcmp_im_.get_R();

        // Death. A simplex τ gives death in Im(g →f ) iff τ is negative in Rf
        // and the lowest one in its column in R_im corresponds to a simplex σ ∈ L.
        // Then (σ, τ ) is a pair.
        for(size_t death_idx = 0 ; death_idx < R_im.size() ; ++death_idx) {
            // tau is positive in R_f -> skip it
            if (dcmp_im_.is_positive(death_idx))
                continue;

            auto im_low = low(R_im.at(death_idx));

            // lowest one in the column of tau in R_im is not in L, skip it
            if (im_low >= static_cast<Int>(fil_L_.size()))
                continue;

            auto birth_idx = new_order_to_old_.at(im_low);

            if (birth_idx == k_invalid_index)
                throw std::runtime_error("indexing error in generate_im_diagrams");

            Real birth = fil_K_.value_by_sorted_id(birth_idx);
            Real death = fil_K_.value_by_sorted_id(death_idx);
            dim_type dim = fil_K_.get_cell(birth_idx).dim();

            if (birth != death or params_.include_zero_persistence)
                im_diagrams_.add_point(dim, birth, death, birth_idx, death_idx, fil_K_.get_id_by_sorted_id(birth_idx), fil_K_.get_id_by_sorted_id(death_idx));

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
                size_t birth_idx = sorted_L_to_sorted_K_.at(sigma_L_idx);
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
                im_diagrams_.add_point(dim, birth, fil_K_.infinity(), birth_idx, plus_inf, fil_K_.get_id_by_sorted_id(birth_idx), plus_inf);
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

    const Dgms& get_domain_diagrams() const
    {
        // these are always computed
        return dom_diagrams_;
    }

    const Dgms& get_codomain_diagrams() const
    {
        if (not params_.codomain)
            throw std::runtime_error("codomain diagrams were not computed because params.cokernel was false in constructor");
        return cod_diagrams_;
    }

};
} // namespace oineus
