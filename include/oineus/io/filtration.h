#pragma once

#include <vector>



template<typename Int_, typename Real_>
struct Simplex_ {
    using IdxVector = std::vector<Int>;

    Int id {-1};
    std::vector<Int> vertices;

    Int dim() const { return static_cast<Int>(vertices.size()) - 1; }

    Real value {-1};

    std::vector<IdxVector> boundary_s() const;
};


template<typename Int_, size_t D>
class Grid {

    using Int = Int_;
    using GridPoint = std::array<Int, D>;
    using PointVec = std::vector<GridPoint>;
    using SimplexVec = std::vector<Simplex>;

    static constexpr size_t dim { D };

    Grid(const GridPoint& _dims, bool wrap_) : dims_(_dims), wrap_(_wrap)
    {
        powers_of_two_[0] = 1;
        for(size_t i = 1; i < powers_of_two_.size(); ++i)
        {
            powers_of_two_[i] = 2 * powers_of_two_[i-1];
        }
    }

    PointVec fr_link(const GridPoint& p) const
    {
        PointVec result;
        result.reserve(powers_of_two_[D] - 1);

        for(Int neighbor_idx = 1; neighbor_idx <= powers_of_two_[D]; ++neighbor_idx)
        {
            GridPoint cand = p;
            bool out_of_dims = false;

            for(size_t c = 0; c < D; ++c)
                if (neighbor_idx & powers_of_two_[c + 1]) {
                    ++cand[c];
                    out_of_dims = out_of_dims or cand[c] >= dims_[c];
                }

            if (out_of_dims and not wrap)
                continue;

            if (wrap and out_of_dims)
                for(size_t c = 0; c < D; ++c)
                    cand[c] = cand[c] % dims[c];

            result.push_back(cand);
        }

        return result;
    }

    SimplexVec fr_simplices(const GridPoint& v, int dim) const
    {
        PointVec vs = fr_link(v);

    }





private:
    GridPoint dims_;
    bool wrap_;
    std::array<Int, D+1> powers_of_two_;
};








template<typename Int_, typename Real_>
class Filtration {
public:
    using Int = Int_;
    using Real = Real_;

    using Simplex = Simplex_<Int_, Real_>;


private:


    std::vector<Simplex> simplices;
    std::map<IdxVector, IdxType> simplex_to_id;

    void clear()
    {
        simplices.clear();
        simplex_to_id.clear();
    }

    void set_ids()
    {
        simplex_to_id.clear();
        int id = 0;
        for(Simplex& s : simplices) {
            s.id = id;
            simplex_to_id[s.vertices] = id;
            ++id;
        }
    }

    IdxType dim_by_id(IdxType id) const { return simplices[id].dim(); }
};

void read_filtration(std::string fname_in, SparseMatrix& r_matrix, Filtration& fil, bool prepare_for_clearing);
void read_phat_boundary_matrix(std::string matrix_fname, SparseMatrix& r_matrix,
        bool prepare_for_clearing);
