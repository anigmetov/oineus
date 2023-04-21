#ifndef OINEUS_VECTORIZER_H
#define OINEUS_VECTORIZER_H

#include <vector>
#include <chrono>
#include <iostream>
#include <cmath>
#include <limits>
#include <functional>
#include <atomic>
#include <thread>
#include <pthread.h>

#include "common_defs.h"

#include "diagram.h"

namespace oineus {

    struct ImageResolution {
        size_t n_pixels_x;
        size_t n_pixels_y;
    };

    template<typename Real>
    struct RImageBounds {
        Real min_x {std::numeric_limits<Real>::quiet_NaN()};
        Real min_y {std::numeric_limits<Real>::quiet_NaN()};
        Real max_x {std::numeric_limits<Real>::quiet_NaN()};
        Real max_y {std::numeric_limits<Real>::quiet_NaN()};

        bool is_defined() const { return not is_undefined(); }

        bool is_undefined() const
        {
            assert((std::isnan(min_x) and std::isnan(min_y) and std::isnan(max_x) and std::isnan(max_y))
                    xor (not std::isnan(min_x) and not std::isnan(min_y) and not std::isnan(max_x)
                            and not std::isnan(max_y)));
            return std::isnan(min_x);
        }
    };

    template<typename Real>
    std::vector<RPoint < Real>>
    transform_rotate_45(const typename Diagrams<Real>::Dgm& dgm)
    {
        std::vector<RPoint<Real>> result;
        result.reserve(dgm.size());
        const Real sq2 = sqrt(2);
        for(const auto& dp: dgm) {

            if (dp.is_inf())
                continue;

            Real x = (dp.birth + dp.death) / sq2;
            Real y = (dp.birth - dp.death) / sq2;

            result.emplace_back(x, y);
        }
        return result;
    }

    template<typename Real>
    std::vector<RPoint < Real>>
    transform_birth_persistence(const typename Diagrams<Real>::Dgm& dgm)
    {
        std::vector<RPoint<Real>> result;
        result.reserve(dgm.size());
        for(const auto& dp: dgm) {

            if (dp.is_inf())
                continue;

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
    template<typename Real>
    Real erf_diff(Real x, Real dx, Real z, Real sigma)
    {
        Real a2 = (x + dx - z) / (sqrt(static_cast<Real>(2)) * sigma);
        Real a1 = (x - z) / (sqrt(static_cast<Real>(2)) * sigma);
        return erf(a2) - erf(a1);
    }

    template<typename Real>
    Real eval_func_persim_unstable(RPoint<Real> eval_point, RPoint<Real> dgm_point, Real dx, Real dy, Real sigma)
    {
        return erf_diff(eval_point.x, dx, dgm_point.x, sigma) * erf_diff(eval_point.y, dy, dgm_point.y, sigma) / 4;
    }

    template<typename Real>
    Real eval_func_persim_dirac_unstable(RPoint<Real> eval_point, RPoint<Real> dgm_point, Real dx, Real dy, Real sigma)
    {
        // take difference from pixel center to diagram point
        Real xdiff = eval_point.x + (dx / 2) - dgm_point.x;
        Real ydiff = eval_point.y + (dy / 2) - dgm_point.y;
        Real sqdist = xdiff * xdiff + ydiff * ydiff;
        return exp(-sqdist / (2 * sigma));
    }

    template<class Real, class F>
    void vectorizer_helper_thread_func(std::vector<std::atomic<Real>>& result,
            const typename std::vector<RPoint < Real>>
    & dgm_points,
    const typename std::vector<RPoint < Real>>& eval_points,
    Real dx, Real
    dy,
    Real sigma, F
    f,
    size_t ep_start,
            size_t
    ep_end) {
    for (
    size_t eval_point_idx = ep_start;
    eval_point_idx<ep_end;
    ++eval_point_idx) {
    Real local_result = 0;

    for (
    const auto& dgm_point
    : dgm_points)
    local_result +=
    f(eval_points[eval_point_idx], dgm_point, dx, dy, sigma
    );

    Real expected, desired;
{
    expected = result[eval_point_idx].load();
    desired = expected + local_result;
}
while (not
std::atomic_compare_exchange_weak_explicit(& result[eval_point_idx],
        & expected,
        desired,
        std::memory_order_relaxed,
        std::memory_order_relaxed
));
}
}

template<typename Real>
class Vectorizer {
public:
    using ImageBounds = RImageBounds<Real>;
    using Point = RPoint<Real>;
    using PointVec = std::vector<Point>;
    using RealVec = std::vector<Real>;
    using Diagram = typename Diagrams<Real>::Dgm;

    enum class TransformType {
        ROTATE,
        BIRTH_PERSISTENCE
    };

    Vectorizer() = default;

    Vectorizer(Real _sigma,
            ImageResolution _resolution = ImageResolution(),
            TransformType _transform_type = TransformType::ROTATE,
            ImageBounds _bounds = ImageBounds())
            :
            sigma_(_sigma),
            resolution_(_resolution),
            transform_type_(_transform_type),
            bounds_(_bounds)
    {
        if (sigma_ <= 0)
            throw std::runtime_error("negative sigma");
    }

    bool bounds_defined() const { return bounds_.is_defined(); }

    void set_verbose(bool _verbose) { verbose_ = _verbose; }

    void set_n_threads(size_t _n_threads) { n_threads_ = _n_threads; }

    size_t get_n_threads() const { return n_threads_; }

    void set_bounds(const PointVec& points)
    {
        if (points.empty())
            return;

        bounds_.min_x = bounds_.min_y = std::numeric_limits<Real>::max();
        bounds_.max_x = bounds_.max_y = std::numeric_limits<Real>::lowest();

        for(const auto& p: points) {
            bounds_.min_x = std::min(p.x, bounds_.min_x);
            bounds_.min_y = std::min(p.y, bounds_.min_y);
            bounds_.max_x = std::max(p.x, bounds_.max_x);
            bounds_.max_y = std::max(p.y, bounds_.max_y);
        }
    }

    PointVec transform(const Diagram& dgm) const
    {
        switch(transform_type_) {
        case TransformType::ROTATE :return transform_rotate_45<Real>(dgm);
        case TransformType::BIRTH_PERSISTENCE :return transform_birth_persistence<Real>(dgm);
        default:throw std::runtime_error("Unknown transformation type");
        }
    }

    RealVec persistence_image_unstable(const Diagram& dgm)
    {
        if (n_threads_ == 1)
            return persistence_image_gen_serial(dgm, eval_func_persim_unstable < Real > );
        else
            return persistence_image_gen_parallel(dgm, eval_func_persim_unstable < Real > );
    }

    RealVec persistence_image_dirac_unstable(const Diagram& dgm)
    {
        if (n_threads_ == 1)
            return persistence_image_gen_serial(dgm, eval_func_persim_dirac_unstable < Real > );
        else
            return persistence_image_gen_parallel(dgm, eval_func_persim_dirac_unstable < Real > );
    }

    // for tests
    RealVec persistence_image_unstable_serial(const Diagram& dgm)
    {
        return persistence_image_gen_serial(dgm, eval_func_persim_unstable < Real > );
    }

    RealVec persistence_image_dirac_unstable_serial(const Diagram& dgm)
    {
        return persistence_image_gen_serial(dgm, eval_func_persim_dirac_unstable < Real > );
    }

private:

    Real sigma_ {1};
    ImageResolution resolution_;
    TransformType transform_type_ {TransformType::ROTATE};
    ImageBounds bounds_;
    bool verbose_ {false};
    size_t n_threads_ {1};

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

        for(size_t i = 0; i < resolution_.n_pixels_x; ++i)
            for(size_t j = 0; j < resolution_.n_pixels_y; ++j)
                lower_left_points.emplace_back(bounds_.min_x + i * get_dx(), bounds_.min_y + j * get_dy());

        return lower_left_points;
    }

    template<class F>
    RealVec persistence_image_gen_parallel(const Diagram& dgm, F f)
    {
        PointVec dgm_points = transform(dgm);

        if (not bounds_defined())
            set_bounds(dgm_points);

        auto eval_points = get_lower_left_points();

        size_t n_eval_points = eval_points.size();
        size_t n_dgm_points = dgm_points.size();

        Real dx = get_dx();
        Real dy = get_dy();

        std::vector<Real> result(n_eval_points, Real(0));

        std::vector<std::atomic<Real>> atomic_result(n_eval_points);
        for(size_t i = 0; i < n_eval_points; ++i)
            atomic_result[i].store(0, std::memory_order_relaxed);

        auto start = std::chrono::steady_clock::now();

        std::vector<std::thread> ts;

        size_t d_ep = n_eval_points / n_threads_;

        for(size_t thread_idx = 0; thread_idx < n_threads_; ++thread_idx) {
            size_t ep_start = thread_idx * d_ep;
            // last thread gets all remaining points
            size_t ep_end = (thread_idx == n_threads_ - 1) ? n_eval_points : (thread_idx + 1) * d_ep;

//            std::cerr << "creating thread " << thread_idx << ", ep_start = " << ep_start << ", ep_end = " << ep_end << std::endl;

            ts.emplace_back(vectorizer_helper_thread_func < Real, F > ,
                    std::ref(atomic_result), std::ref(dgm_points), std::ref(eval_points),
                    dx, dy, sigma_, f, ep_start, ep_end);

#ifdef __linux__
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(thread_idx, &cpuset);
            int rc = pthread_setaffinity_np(ts[thread_idx].native_handle(), sizeof(cpu_set_t), &cpuset);
            if (rc != 0) { std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n"; }
#endif
        }

        for(auto& t: ts) {
            t.join();
        }

        for(size_t result_idx = 0; result_idx < n_eval_points; ++result_idx)
            result[result_idx] = atomic_result[result_idx].load(std::memory_order_relaxed);

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        if (verbose_)
            std::cerr << "diagram points: " << n_dgm_points << ", n_threads = " << n_threads_ << ", pers. image time = "
                      << elapsed.count() << std::endl;

        return result;
    }

    template<class F>
    RealVec persistence_image_gen_serial(const Diagram& dgm, F f)
    {
        PointVec dgm_points = transform(dgm);

        if (not bounds_defined())
            set_bounds(dgm_points);

        auto eval_points = get_lower_left_points();

        size_t n_eval_points = eval_points.size();
        size_t n_dgm_points = dgm_points.size();

        Real dx = get_dx();
        Real dy = get_dy();

        std::vector<Real> result(n_eval_points, Real(0));

        auto start = std::chrono::steady_clock::now();

        for(size_t dgm_point_idx = 0; dgm_point_idx < n_dgm_points; ++dgm_point_idx)
            for(size_t eval_point_idx = 0; eval_point_idx < n_eval_points; ++eval_point_idx)
                result[eval_point_idx] += f(eval_points[eval_point_idx], dgm_points[dgm_point_idx],
                        dx, dy,
                        sigma_);

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        if (verbose_)
            std::cerr << "diagram points: " << n_dgm_points << ", pers. image time = " << elapsed.count() << std::endl;

        return result;
    }
}; // Vectorizer

} // namespace oineus

#endif //OINEUS_VECTORIZER_H