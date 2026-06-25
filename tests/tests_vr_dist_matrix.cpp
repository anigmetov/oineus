// Regression test for Vietoris-Rips construction from a pairwise-distance
// matrix vs a point cloud, across every construction path:
//
//   algorithm : VRE (in-order, get_vr_filtration[_and_critical_edges]_inorder)
//               BK  (Bron-Kerbosch, get_vr_filtration[_and_critical_edges])
//   input     : point cloud  /  full pairwise-distance matrix (DistMatrix)
//   edges     : with / without critical (longest) edges
//   threshold : unbounded (full complex) / finite cutoff (subset)
//
// The convention under test (matching Ripser, Gudhi, Dionysus, Dipha): the
// distance matrix holds ACTUAL distances and max_diameter is an actual-distance
// threshold. A historical bug compared dist <= max_diameter*max_diameter in the
// DistMatrix Bron-Kerbosch paths, i.e. it silently treated the cutoff as a
// squared distance and dropped most edges.
//
// The 5-point cloud sits on a line with pairwise distances
// {2,3,4,5,6,7,9,10,13,15} -- all distinct and NONE equal to 1, so a squared
// vs. actual mix-up cannot hide behind 1*1 == 1. All expected filtration values
// and longest edges are hard-coded below.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>
#include <vector>

#include <oineus/oineus.h>

namespace {

using Real = double;
using Int = int;
using oineus::dim_type;
using Point1 = oineus::Point<Real, 1>;
using VKey = std::vector<Int>;          // sorted vertex set, a canonical cell key
using Edge = std::pair<Int, Int>;       // normalized {min, max}

// 5 collinear points; pairwise distances are distinct integers, none == 1.
const std::vector<Point1> kPoints = {{0.0}, {2.0}, {5.0}, {9.0}, {15.0}};

// Row-major 5x5 distance matrix, D[i*5 + j] = |x_i - x_j|.
const std::vector<Real> kDist = {
     0,  2,  5,  9, 15,
     2,  0,  3,  7, 13,
     5,  3,  0,  4, 10,
     9,  7,  4,  0,  6,
    15, 13, 10,  6,  0,
};

// Expected filtration value (= simplex diameter) for the FULL complex.
const std::map<VKey, Real> kExpectedFull = {
    {{0}, 0}, {{1}, 0}, {{2}, 0}, {{3}, 0}, {{4}, 0},
    {{0, 1}, 2}, {{0, 2}, 5}, {{0, 3}, 9}, {{0, 4}, 15},
    {{1, 2}, 3}, {{1, 3}, 7}, {{1, 4}, 13},
    {{2, 3}, 4}, {{2, 4}, 10}, {{3, 4}, 6},
    {{0, 1, 2}, 5}, {{0, 1, 3}, 9}, {{0, 1, 4}, 15}, {{0, 2, 3}, 9},
    {{0, 2, 4}, 15}, {{0, 3, 4}, 15}, {{1, 2, 3}, 7}, {{1, 2, 4}, 13},
    {{1, 3, 4}, 13}, {{2, 3, 4}, 10},
};

// Expected filtration with a finite cutoff max_diameter = 8: keep only simplices
// whose diameter <= 8 (6 edges + 2 triangles survive).
const std::map<VKey, Real> kExpectedThr8 = {
    {{0}, 0}, {{1}, 0}, {{2}, 0}, {{3}, 0}, {{4}, 0},
    {{0, 1}, 2}, {{0, 2}, 5}, {{1, 2}, 3}, {{1, 3}, 7}, {{2, 3}, 4}, {{3, 4}, 6},
    {{0, 1, 2}, 5}, {{1, 2, 3}, 7},
};

// Expected longest (critical) edge per simplex of dim >= 1 (superset; the
// thresholded complex looks up only the cells it contains).
const std::map<VKey, Edge> kExpectedEdge = {
    {{0, 1}, {0, 1}}, {{0, 2}, {0, 2}}, {{0, 3}, {0, 3}}, {{0, 4}, {0, 4}},
    {{1, 2}, {1, 2}}, {{1, 3}, {1, 3}}, {{1, 4}, {1, 4}},
    {{2, 3}, {2, 3}}, {{2, 4}, {2, 4}}, {{3, 4}, {3, 4}},
    {{0, 1, 2}, {0, 2}}, {{0, 1, 3}, {0, 3}}, {{0, 1, 4}, {0, 4}},
    {{0, 2, 3}, {0, 3}}, {{0, 2, 4}, {0, 4}}, {{0, 3, 4}, {0, 4}},
    {{1, 2, 3}, {1, 3}}, {{1, 2, 4}, {1, 4}}, {{1, 3, 4}, {1, 4}},
    {{2, 3, 4}, {2, 4}},
};

oineus::DistMatrix<Real> dist_matrix()
{
    return oineus::DistMatrix<Real>{kDist.data(), 5};
}

VKey cell_key(const oineus::CellWithValue<oineus::Simplex<Int>, Real>& c)
{
    VKey vs(c.get_cell().get_vertices().begin(), c.get_cell().get_vertices().end());
    std::sort(vs.begin(), vs.end());
    return vs;
}

template<class Fil>
void check_values(const Fil& fil, const std::map<VKey, Real>& expected)
{
    std::map<VKey, Real> got;
    for (size_t i = 0; i < fil.size(); ++i) {
        const auto& c = fil.get_cell(i);
        got[cell_key(c)] = c.get_value();
    }
    REQUIRE(got.size() == expected.size());
    for (const auto& [key, val] : expected) {
        REQUIRE(got.count(key) == 1);
        REQUIRE(std::abs(got.at(key) - val) < Real(1e-12));
    }
}

template<class Fil>
void check_edges(const Fil& fil, const std::vector<oineus::VREdge<Int>>& edges)
{
    REQUIRE(fil.size() == edges.size());
    auto dm = dist_matrix();
    for (size_t i = 0; i < fil.size(); ++i) {
        const auto& c = fil.get_cell(i);
        if (c.dim() == 0)
            continue;
        VKey key = cell_key(c);
        Edge e{std::min(edges[i].x, edges[i].y), std::max(edges[i].x, edges[i].y)};
        REQUIRE(kExpectedEdge.count(key) == 1);
        REQUIRE(e == kExpectedEdge.at(key));
        // the critical edge's length must equal the simplex's filtration value
        REQUIRE(std::abs(dm.get_distance(e.first, e.second) - c.get_value()) < Real(1e-12));
    }
}

}  // namespace

TEST_CASE("VR distance-matrix and point-cloud paths agree with hard-coded ground truth")
{
    const dim_type max_dim = 2;

    struct Case { Real radius; const std::map<VKey, Real>* expected; const char* name; };
    const std::vector<Case> cases = {
        {100.0, &kExpectedFull, "unbounded"},
        {8.0,   &kExpectedThr8, "threshold=8"},
    };

    for (const auto& tc : cases) {
        const Real r = tc.radius;
        auto dm = dist_matrix();

        SECTION(std::string("no critical edges -- ") + tc.name) {
            // VRE
            check_values(oineus::get_vr_filtration_inorder<Int, Real, 1>(kPoints, max_dim, r), *tc.expected);
            check_values(oineus::get_vr_filtration_inorder<Int, Real>(dm, max_dim, r), *tc.expected);
            // Bron-Kerbosch
            check_values(oineus::get_vr_filtration<Int, Real, 1>(kPoints, max_dim, r), *tc.expected);
            check_values(oineus::get_vr_filtration<Int, Real>(dm, max_dim, r), *tc.expected);
        }

        SECTION(std::string("with critical edges -- ") + tc.name) {
            {
                auto [f, e] = oineus::get_vr_filtration_and_critical_edges_inorder<Int, Real, 1>(kPoints, max_dim, r);
                check_values(f, *tc.expected);
                check_edges(f, e);
            }
            {
                auto [f, e] = oineus::get_vr_filtration_and_critical_edges_inorder<Int, Real>(dm, max_dim, r);
                check_values(f, *tc.expected);
                check_edges(f, e);
            }
            {
                auto [f, e] = oineus::get_vr_filtration_and_critical_edges<Int, Real, 1>(kPoints, max_dim, r);
                check_values(f, *tc.expected);
                check_edges(f, e);
            }
            {
                auto [f, e] = oineus::get_vr_filtration_and_critical_edges<Int, Real>(dm, max_dim, r);
                check_values(f, *tc.expected);
                check_edges(f, e);
            }
        }
    }
}
