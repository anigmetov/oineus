
template<class Real>
std::vector<Real> persfunc_unstable(py::array_t<Real> points, py::array_t<Real> diagram, Real sigma)
{
    bool verbose = false;

    auto eval_points = points.template unchecked<2>();
    auto dgm_points = diagram.template unchecked<2>();

    if (eval_points.shape(1) != 2) {
        throw std::runtime_error("points must be an Nx2 NumPy array");
    }

    if (dgm_points.shape(1) != 2) {
        throw std::runtime_error("diagram must be an Nx2 NumPy array");
    }

    size_t n_eval_points = eval_points.shape(0);
    size_t n_dgm_points = dgm_points.shape(0);

    std::vector<Real> result(n_eval_points, Real(0));

    auto start = std::chrono::steady_clock::now();
    for(size_t eval_point_idx = 0; eval_point_idx < n_eval_points; ++eval_point_idx) {

        for(size_t dgm_point_idx = 0; dgm_point_idx < n_dgm_points; ++dgm_point_idx) {

            Real xdiff = eval_points(eval_point_idx, 0) - dgm_points(dgm_point_idx, 0);
            Real ydiff = eval_points(eval_point_idx, 1) - dgm_points(dgm_point_idx, 1);
            Real sqdist = xdiff * xdiff + ydiff * ydiff;

            result[eval_point_idx] += exp(-sqdist / sigma);
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    if (verbose)
        std::cerr << "diagram points: " << n_dgm_points << ", smilis time = " << elapsed.count() << std::endl;

    return result;
}
