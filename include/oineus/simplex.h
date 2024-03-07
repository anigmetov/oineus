#pragma once

#include <limits>
#include <vector>
#include <ostream>
#include <algorithm>
#include <utility>
#include <unordered_set>
#include <string>
#include <sstream>

//using namespace std::rel_ops;

namespace oineus {

struct VREdge {
    size_t x;
    size_t y;
    bool operator==(const VREdge& other) const { return x == other.x and y == other.y; }
    bool operator!=(const VREdge& other) const { return !(*this == other); }
    bool operator<(const VREdge& other) const { return x < other.x or (x == other.x and y < other.y); };
    bool operator>(const VREdge& other) const { return other < *this; }
    bool operator<=(const VREdge& other) const { return *this < other or *this == other; }
    bool operator>=(const VREdge& other) const { return *this > other or *this == other; }
};

inline std::ostream& operator<<(std::ostream& out, const VREdge& e)
{
    out << "edge(x=" << e.x << ", y=" << e.y << ")";
    return out;
}

template<typename Int_>
struct Simplex {
    using Int = Int_;
    using IdxVector = std::vector<Int>;

    using Uid = IdxVector;
    // for Z2 only for now
    using Boundary = std::vector<Uid>;

    static constexpr Int k_invalid_id = Int(-1);

    Int id_ {k_invalid_id};
    IdxVector vertices_;

    Simplex() = default;
    Simplex(const Simplex&) = default;
    Simplex(Simplex&&) = default;
    Simplex& operator=(const Simplex&) = default;
    Simplex& operator=(Simplex&&) = default;

    Simplex(const IdxVector& _vertices)
            :vertices_(_vertices)
    {
        if (vertices_.empty())
            throw std::runtime_error("Empty simplex not allowed");

        if (vertices_.size() == 1)
            id_ = vertices_[0];
        else
            std::sort(vertices_.begin(), vertices_.end());
    }

    dim_type dim() const { return static_cast<Int>(vertices_.size()) - 1; }

    Int get_id() const { return id_; }
    void set_id(Int new_id) { id_ = new_id; }

    Simplex(const int _id, const IdxVector& _vertices)
            :vertices_(_vertices), id_(_id)
    {
        if (vertices_.empty())
            throw std::runtime_error("Empty simplex not allowed");

        if (vertices_.size() > 1)
            std::sort(vertices_.begin(), vertices_.end());
    }

    Boundary boundary() const
    {
        std::vector<IdxVector> bdry;

        if (dim() == 0)
            return bdry;

        bdry.reserve(vertices_.size());

        for(size_t i = 0 ; i < vertices_.size() ; ++i) {
            IdxVector tau;
            tau.reserve(vertices_.size() - 1);

            for(size_t j = 0 ; j < vertices_.size() ; ++j)
                if (j != i)
                    tau.push_back(vertices_[j]);

            // vertices_ is sorted -> tau is sorted automatically

            bdry.push_back(tau);
        }

        return bdry;
    }

    // create a new simplex by joining with vertex and assign value to it
    Simplex join(Int new_id, Int vertex) const
    {
        // new vertex must not be present in this->vertices
        assert(std::find(vertices_.begin(), vertices_.end(), vertex) == vertices_.end());

        IdxVector vs = vertices_;
        vs.push_back(vertex);
        return Simplex(new_id, vs);
    }

    Simplex join(Int vertex)
    {
        return join(k_invalid_id, vertex);
    }

    bool operator==(const Simplex& other) const
    {
        // ignore id_ ?
        return get_id() == other.get_id() and vertices_ == other.vertices_;
    }

    bool operator!=(const Simplex& other) const
    {
        return !(*this == other);
    }

    template<typename I>
    friend std::ostream& operator<<(std::ostream&, const Simplex<I>&);

    const Uid& get_uid() const { return vertices_; }
    void set_uid(const Uid& new_vs)
    {
        vertices_ = new_vs;
        std::sort(vertices_.begin(), vertices_.end());
    }

    static std::string uid_to_string(const Uid& uid)
    {
        std::stringstream ss;
        ss << "[";
        for(auto v : uid) {
            ss << v << ",";
        }
        ss << "]";
        return ss.str();
    }

    std::string uid_as_string() const
    {
        return uid_to_string(get_uid());
    }

    struct UidHasher {
        std::size_t operator()(const Uid& vs) const
        {
            // TODO: replace with better hash function
            std::size_t seed = 0;
            for(auto v: vs)
                oineus::hash_combine(seed, v);
            return seed;
        }
    };

    std::string repr() const
    {
        std::stringstream out;
        out << "Simplex(id_=" << id_ << ", vertices_=[";

        for(size_t i = 0 ; i < vertices_.size() - 1 ; ++i)
            out << vertices_[i] << ", ";

        out << vertices_[vertices_.size() - 1] << "])";

        return out.str();
    }

    using UidSet = std::unordered_set<Uid, UidHasher>;
};

template<typename I>
std::ostream& operator<<(std::ostream& out, const Simplex<I>& s)
{
    out << "[";

    for(size_t i = 0 ; i < s.vertices_.size() - 1 ; ++i)
        out << s.vertices_[i] << ", ";

    out << s.vertices_[s.vertices_.size() - 1] << "]";

    return out;
}
}

namespace std {
template<>
struct hash<oineus::VREdge> {
    std::size_t operator()(const oineus::VREdge& p) const
    {
        std::size_t seed = 0;
        oineus::hash_combine(seed, p.x);
        oineus::hash_combine(seed, p.y);
        return seed;
    }
};
};
