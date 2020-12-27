#ifndef OINEUS_VECTORIZER_H
#define OINEUS_VECTORIZER_H

#include <vector>
#include <chrono>
#include <iostream>
#include <cmath>
#include <limits>

#include "common_defs.h"
#include "diagram.h"

namespace oineus {

struct ImageResolution {
    size_t n_pixels_x;
    size_t n_pixels_y;
};

template <typename Real>
struct RImageBounds {
    Real min_x{std::numeric_limits<Real>::quiet_NaN()};
    Real min_y{std::numeric_limits<Real>::quiet_NaN()};
    Real max_x{std::numeric_limits<Real>::quiet_NaN()};
    Real max_y{std::numeric_limits<Real>::quiet_NaN()};

    bool is_defined() const { return not is_undefined(); }

    bool is_undefined() const
    {
        assert((std::isnan(min_x) and
                std::isnan(min_y) and
                std::isnan(max_x) and
                std::isnan(max_y))
                xor (not std::isnan(min_x) and
                        not std::isnan(min_y) and
                        not std::isnan(max_x) and
                        not std::isnan(max_y)));
        return std::isnan(min_x);
    }
};

template <typename Real>
std::vector<RPoint<Real>> transform_rotate_45(const typename Diagram<Real>::Dgm& dgm)
{
    std::vector<RPoint<Real>> result;
    result.reserve(dgm.size());
    constexpr Real sq2 = sqrt(2);
    for (const auto& dp : dgm) {

        if (dp.is_infinite())
            continue;

        Real x = (dp.birth + dp.death) / sq2;
        Real y = (dp.birth - dp.death) / sq2;

        result.emplace_back(x, y);
    }
    return result;
}

template <typename Real>
std::vector<RPoint<Real>> transform_birth_persistence(const typename Diagram<Real>::Dgm& dgm)
{
    std::vector<RPoint<Real>> result;
    result.reserve(dgm.size());
    for (const auto& dp : dgm) {
        Real x = dp.birth;
        Real y = (dp.death - dp.birth);
        result.emplace_back(x, y);
    }
    return result;
}

// helper function to use in persistence_image
// x: coordinate of the lower-left corner of pixel
// dx: size of pixel (along the corresponding axis)
// z: coordinate of the diagram point (after transformation)
// sigma: standard deviation of Gaussian
template <typename Real>
Real erf_diff(Real x, Real dx, Real z, Real sigma)
{
    Real a2 = (x + dx - z) / (sqrt(static_cast<Real>(2)) * sigma);
    Real a1 = (x - z) / (sqrt(static_cast<Real>(2)) * sigma);
    return erf(a2) - erf(a1);
}

template <typename Real, typename EP, typename DP>
class Vectorizer {
public:
    using ImageBounds = RImageBounds<Real>;
    using Point = RPoint<Real>;
    using PointVec = std::vector<Point>;
    using RealVec = std::vector<Real>;

    Vectorizer() = default;

    Vectorizer(Real sigma_, ImageBounds _bounds, ImageResolution _resolution = ImageResolution());

private:

    Real sigma_;

    ImageResolution resolution_;
    ImageBounds bounds_;

    Real get_dx() const
    {
        assert(bounds_.is_defined());
        return (bounds_.max_x - bounds_.min_x) / resolution_.n_pixels_x;
    }

    Real get_dy() const
    {
        assert(bounds_.is_defined());
        return (bounds_.max_y - bounds_.min_y) / resolution_.n_pixels_y;
    }

    PointVec get_lower_left_points() const
    {
        PointVec lower_left_points;
        for (size_t i = 0; i < resolution_.n_pixels_x; ++i)
            for (size_t j = 0; j < resolution_.n_pixels_y; ++j)
                lower_left_points.emplace_back(bounds_.min_x + i * get_dx(), bounds_.min_y + j * get_dy());
        return lower_left_points;
    }

    PointVec get_center_points() const
    {
        PointVec center_points;
        for (size_t i = 0; i < resolution_.n_pixels_x; ++i)
            for (size_t j = 0; j < resolution_.n_pixels_y; ++j)
                center_points.emplace_back(bounds_.min_x + (i + Real(0.5)) * get_dx(),
                        bounds_.min_y + (j + Real(0.5)) * get_dy());
        return center_points;
    }

    RealVec persistence_image_unstable(const PointVec& dgm_points, Real sigma)
    {
        bool verbose = false;

        size_t n_eval_points = lower_left_points.size();
        size_t n_dgm_points = dgm_points.size();

        std::vector<Real> result(n_eval_points, Real(0));

        auto start = std::chrono::steady_clock::now();

        for (size_t eval_point_idx = 0; eval_point_idx < n_eval_points; ++eval_point_idx) {

            Real ep_x = lower_left_points[eval_point_idx][0];
            Real ep_y = lower_left_points[eval_point_idx][1];

            for (size_t dgm_point_idx = 0; dgm_point_idx < n_dgm_points; ++dgm_point_idx) {

                Real dgm_x = dgm_points[dgm_point_idx][0];
                Real dgm_y = dgm_points[dgm_point_idx][1];

                result[eval_point_idx] += erf_diff(ep_x, dx, dgm_x, sigma) * erf_diff(ep_y, dy, dgm_y, sigma) / 4;
            }
        }

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        if (verbose)
            std::cerr << "diagram points: " << n_dgm_points << ", pers. image time = " << elapsed.count() << std::endl;

        return result;
    }

// EP: eval_points type, callable
    template <typename Real, typename EP, typename DP>
    std::vector<Real> persistence_image_dirac_unstable(const EP& eval_points, const DP& dgm_points, Real sigma)
    {
        bool verbose = false;

        size_t n_eval_points = eval_points.size();
        size_t n_dgm_points = dgm_points.size();

        std::vector<Real> result(n_eval_points, Real(0));

        auto start = std::chrono::steady_clock::now();

        for (size_t eval_point_idx = 0; eval_point_idx < n_eval_points; ++eval_point_idx) {
            for (size_t dgm_point_idx = 0; dgm_point_idx < n_dgm_points; ++dgm_point_idx) {
                Real xdiff = eval_points(eval_point_idx, 0) - dgm_points(dgm_point_idx, 0);
                Real ydiff = eval_points(eval_point_idx, 1) - dgm_points(dgm_point_idx, 1);
                Real sqdist = xdiff * xdiff + ydiff * ydiff;
                result[eval_point_idx] += exp(-sqdist / sigma);
            }
        }

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        if (verbose)
            std::cerr << "diagram points: " << n_dgm_points << ", pers. image time = " << elapsed.count() << std::endl;

        return result;
    }


};

#endif //OINEUS_VECTORIZER_H