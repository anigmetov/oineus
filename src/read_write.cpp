#include <iostream>
#include <string>
#include <map>
#include <iterator>
#include <algorithm>

#include "common_defs.h"
#include "read_write.h"

namespace pp {

    std::vector<std::vector<IdxType>> Simplex::boundary_s() const
    {

        std::vector<IdxVector> result;

        if (dim() > 0) {
            for(size_t i = 0; i < vertices.size(); ++i) {
                IdxVector bdry_spx;
                for(size_t j = 0; j < vertices.size(); ++j) {
                    if (j != i) {
                        bdry_spx.push_back(vertices[j]);
                    }
                }
                result.push_back(bdry_spx);
            }
        }

        return result;
    }

    Simplex parse_string(std::string s)
    {
        // s = "<3,4> 0.3"
        auto simplex_end = s.find(">");
        std::string s_vertices = s.substr(1, simplex_end - 1); // s_bdry = "3,4"
        std::string s_value = s.substr(simplex_end + 1);  // s_value = " 0.3"

        auto vert_vector = split_by_delim(s_vertices, ',');

        Simplex simplex;

        for(auto b: vert_vector) {
            simplex.vertices.push_back(std::atoi(b.c_str()));
        }
        std::sort(simplex.vertices.begin(), simplex.vertices.end());

        simplex.value = std::atof(s_value.c_str());

        return simplex;
    }

    void rearrange_by_dimension(Filtration& fil_in, Filtration& fil_out)
    {
        fil_out.clear();

        std::vector<int> dim_change_indices;
        const auto& simplices_in = fil_in.simplices;

        auto prev_dim = simplices_in[0].dim();

        dim_change_indices.push_back(0);

        for(size_t i = 1; i < simplices_in.size(); ++i) {
            if (simplices_in[i].dim() > prev_dim) {
                dim_change_indices.push_back(i);
                prev_dim = simplices_in[i].dim();
            }
        }

        dim_change_indices.push_back(simplices_in.size());

        info("dim_change_indices = {}", container_to_string(dim_change_indices));

        for(int i = dim_change_indices.size() - 2; i >= 0; i--) {
            for(int j = dim_change_indices[i]; j < dim_change_indices[i + 1]; ++j) {
                fil_out.simplices.push_back(std::move(fil_in.simplices[j]));
            }
            fil_in.simplices.resize(dim_change_indices[i]);
        }
    }

    template<class M>
    void rearrange_bdry_matrix_by_dimension(M& m_in)
    {
        M m_out;
        std::vector<int> dim_change_indices;
        auto prev_dim = m_in[0].size();
        dim_change_indices.push_back(0);
        for(size_t i = 1; i < m_in.size(); ++i) {
            if (m_in[i].size() > prev_dim) {
                dim_change_indices.push_back(i);
                prev_dim = m_in[i].size();
            }
        }
        dim_change_indices.push_back(m_in.size());
        info("dim_change_indices = {}", container_to_string(dim_change_indices));
        for(int i = dim_change_indices.size() - 2; i >= 0; i--) {
            for(int j = dim_change_indices[i]; j < dim_change_indices[i + 1]; ++j) {
                m_out.push_back(std::move(m_in[j]));
            }
            m_in.resize(dim_change_indices[i]);
        }
        m_in = std::move(m_out);
    }

    void read_phat_boundary_matrix(std::string matrix_fname, SparseMatrix& r_matrix, bool prepare_for_clearing)
    {
        std::map<int, SparseMatrix> dim_to_cols;

        std::ifstream f(matrix_fname);
        if (!f.good()) {
            info("Cannot read D matrix from {}.", matrix_fname);
            throw std::runtime_error("file not found");
        }

        std::string s;

        long row_num = 0;
        while(std::getline(f, s)) {
            row_num++;
            SparseColumn col;
            std::stringstream ss(s);
            size_t simplex_dim;
            IdxType i;
            ss >> simplex_dim;
            while(ss >> i) {
                col.push_back(i);
            }

            if (simplex_dim != col.size()) {
                info("Error in file {}, row {}, dimension = {}, boundary has {} elements",
                        matrix_fname, row_num, simplex_dim, col.size());
                throw std::runtime_error("error in boundary matrix file");
            }

            dim_to_cols[simplex_dim].push_back(col);
        }

        f.close();

        r_matrix.clear();
        r_matrix.reserve(row_num);

        if (prepare_for_clearing) {
            // higher dimensions first
            for(auto d = dim_to_cols.rbegin(); d != dim_to_cols.rend(); ++d) {
                std::move(d->second.begin(), d->second.end(), std::back_inserter(r_matrix));
                d->second.clear();
            }
        } else {
            // standard order
            for(auto d = dim_to_cols.begin(); d != dim_to_cols.end(); ++d) {
                std::move(d->second.begin(), d->second.end(), std::back_inserter(r_matrix));
                d->second.clear();
            }
        }
    }

    void read_filtration(std::string fname_in, SparseMatrix& r_matrix, Filtration& fil, bool prepare_for_clearing)
    {

        std::ifstream f {fname_in.c_str()};

        if (!f.good()) {
            std::cerr << "Cannot open file " << fname_in << std::endl;
            throw std::runtime_error("Cannot open file");
        }

        r_matrix.clear();
        fil.clear();

        std::string s;
        while(std::getline(f, s)) {
            fil.simplices.emplace_back(parse_string(s));
        }

        info("fil read from {}", fname_in);

        std::stable_sort(fil.simplices.begin(), fil.simplices.end(),
                [](const Simplex& a, const Simplex& b) { return a.dim() < b.dim(); });

        info("filtration sorted by dimension");

        if (prepare_for_clearing) {
            Filtration fil_copy = fil;
            info("fil_copy created");
            rearrange_by_dimension(fil_copy, fil);
        }

        info("filtration rearranged by dimension");

        fil.set_ids();

        info("filtration ids set");

        std::string bdry_fname = fname_in + (prepare_for_clearing ? ".clr.bdr" : ".bdr");

        if (!read_bdry_matrix<SparseMatrix, SparseColumn>(bdry_fname, r_matrix)) {
            info("creating boundary matrix");
            for(Simplex& s: fil.simplices) {
                SparseColumn col;
                for(const IdxVector& bdry_s: s.boundary_s()) {
                    col.push_back(fil.simplex_to_id.at(bdry_s));
                }
                std::sort(col.begin(), col.end());
                r_matrix.push_back(col);
            }

            info("created boundary matrix");
            write_bdry_matrix(r_matrix, bdry_fname);
            info("saved boundary matrix");
        }

        debug("read fil = {}", fil);
        debug("read r_matrix = {}", r_matrix);
    } // read_filtration



    void write_diagrams(const Diagram& dgm, const IndexDiagram& i_dgm, std::string fname, bool sort)
    {
        write_diagrams(dgm, fname, ".dgm", 5, sort);
        write_diagrams(i_dgm, fname, ".idx.dgm", 0, sort);
    }

}// namespace pp

namespace std {
    ostream& operator<<(ostream& os, const pp::SparseColumn& col)
    {
        os << pp::container_to_string(col);
        return os;
    }

    ostream& operator<<(ostream& os, const pp::SparseMatrix& m)
    {
        os << "Matrix(\n";
        for(size_t i = 0; i < m.size(); ++i) {
            os << "    " << i << " -> " << m[i] << "\n";
        }
        os << ")\n";
        return os;
    }

    ostream& operator<<(ostream& os, const pp::Simplex& sim)
    {
        os << "Simplex{ id = " << sim.id;
        os << ", value = " << sim.value;
        os << ", vertices = [";
        for(size_t i = 0; i < sim.vertices.size(); ++i) {
            os << sim.vertices[i];
            if (i + 1 < sim.vertices.size()) {
                os << ", ";
            }
        }
        os << "]}";
        return os;
    }

    ostream& operator<<(ostream& os, const pp::Filtration& fil)
    {
        os << "Filtration[\n";
        for(pp::Simplex s: fil.simplices) {
            os << "    " << s << "\n";
        }
        os << "]\n";
        return os;
    }

    void read_phat_boundary_matrix(std::string matrix_fname, SparseMatrix& r_matrix, bool prepare_for_clearing)
    {
        std::map<int, SparseMatrix> dim_to_cols;

        std::ifstream f(matrix_fname);
        if (!f.good()) {
            info("Cannot read D matrix from {}.", matrix_fname);
            throw std::runtime_error("file not found");
        }


        std::string s;

        long row_num = 0;
        while(std::getline(f, s)) {
            row_num++;
            SparseColumn col;
            std::stringstream ss(s);
            size_t simplex_dim;
            IdxType i;
            ss >> simplex_dim;
            while(ss >> i) {
                col.push_back(i);
            }

            if (simplex_dim != col.size()) {
                info("Error in file {}, row {}, dimension = {}, boundary has {} elements",
                        matrix_fname, row_num, simplex_dim, col.size());
                throw std::runtime_error("error in boundary matrix file");
            }

            dim_to_cols[simplex_dim].push_back(col);
        }

        f.close();

        r_matrix.clear();
        r_matrix.reserve(row_num);

        if (prepare_for_clearing) {
            // higher dimensions first
            for(auto d = dim_to_cols.rbegin(); d != dim_to_cols.rend(); ++d) {
                std::move(d->second.begin(), d->second.end(), std::back_inserter(r_matrix));
                d->second.clear();
            }
        } else {
            // standard order
            for(auto d = dim_to_cols.begin(); d != dim_to_cols.end(); ++d) {
                std::move(d->second.begin(), d->second.end(), std::back_inserter(r_matrix));
                d->second.clear();
            }
        }
    }
}
