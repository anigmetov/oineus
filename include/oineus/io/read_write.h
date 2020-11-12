#pragma once

#include <string>
#include <fstream>
#include <ostream>
#include <sstream>
#include <vector>
#include <map>
#include <stdlib.h>

#include "common_defs.h"

namespace pp {

    using IndexDiagramPoint = std::pair<IdxType, IdxType>;
    using DiagramPoint = std::pair<Real, Real>;

    using IndexDiagram = std::map<IdxType, std::vector<IndexDiagramPoint>>;
    using Diagram = std::map<IdxType, std::vector<DiagramPoint>>;

    template<class Cont>
    std::string container_to_string(const Cont& v)
    {
        std::stringstream ss;
        ss << "[";
        for(auto x_iter = v.begin(); x_iter != v.end(); ) {
            ss << *x_iter;
            x_iter = std::next(x_iter);
            if (x_iter != v.end())
                ss << ", ";
        }
//        for(const auto& x : v) {
//            ss << x << ", ";
//        }
        ss << "]";
        return ss.str();
    }

   using IdxVector = std::vector<IdxType>;

    struct Simplex {
        IdxType id {-1};
        std::vector<IdxType> vertices;

        IdxType dim() const { return static_cast<IdxType>(vertices.size()) - 1; }

        Real value {-1};

        std::vector<IdxVector> boundary_s() const;
    };

    struct Filtration {
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

    template<class MatrType>
    inline void write_bdry_matrix(const MatrType& r_matrix, std::string fname)
    {
        std::ofstream f(fname);
        if (!f.good()) {
            std::cerr << "Cannot write to " << fname << std::endl;
            throw std::runtime_error("cannot write to file");

        }
        for(size_t i = 0; i < r_matrix.size(); ++i) {
            f << i;
            if (r_matrix[i].size()) {
                f << " ";
                for(size_t j = 0; j < r_matrix[i].size(); ++j) {
                    f << r_matrix[i][j];
                    if (j + 1 < r_matrix[i].size())
                        f << " ";
                }
            }
            f << "\n";
        }
        f.close();
    }

    template<class MatrType, class ColType>
    inline bool read_bdry_matrix(std::string fname, MatrType& r)
    {
        r.clear();
        std::ifstream f(fname);
        if (!f.good()) {
            info("Cannot read D matrix from {}, it will be computed and saved", fname);
            return false;
        }

        std::string s;

        while(std::getline(f, s)) {
            ColType col;
            std::stringstream ss(s);
            IdxType i;
            ss >> i;
            if (i != (IdxType) r.size()) {
                throw std::runtime_error("index mismatch");
            }
            while(ss >> i) {
                col.push_back(i);
            }
            r.push_back(col);
        }

        f.close();

        return true;
    }

    template<class D>
    inline void write_diagrams(const D& dgm, std::string fname, std::string extension, int prec, bool sort)
    {

        for(const auto& dim_points : dgm) {

            auto points = dim_points.second;

            if (points.empty())
                continue;

            std::sort(points.begin(), points.end());

            auto dim = dim_points.first;

            std::string fname_dim = fname + "." + std::to_string(dim) + extension;

            std::ofstream f(fname_dim);
            if (!f.good()) {
                std::cerr << "Cannot open file " << fname_dim << std::endl;
                throw std::runtime_error("Cannot write diagram");
            }

            f.precision(prec);

            for(auto p : points) {
                f << p.first << " ";
                if (p.second != std::numeric_limits<decltype(p.second)>::max()) {
                    f << p.second << "\n";
                } else {
                    f << "inf" << "\n";
                }
            }
            f.close();

            if (!sort)
                continue;

            std::string cmd = "sort -o " + fname_dim + " " + fname_dim;
            if (system(cmd.c_str()) != 0) {
                std::cerr << "Error: sort call failed" << std::endl;
            };
        }

    }

    void write_diagrams(const Diagram& d, const IndexDiagram& id, std::string fname, bool sort);

} // namespace pp

namespace std {
    ostream& operator<<(ostream& os, const pp::SparseColumn& col);
    ostream& operator<<(ostream& os, const pp::SparseMatrix& m);
    ostream& operator<<(ostream& os, const pp::Simplex& sim);
    ostream& operator<<(ostream& os, const pp::Filtration& fil);
}
