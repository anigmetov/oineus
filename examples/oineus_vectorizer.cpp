#include <iostream>
#include <random>

#include <oineus/oineus.h>
#include <opts/opts.h>

int main(int argc, char** argv)
{
    using Real = double;
    using Vectorizer = oineus::Vectorizer<Real>;
    using Dgm = oineus::Diagrams<Real>::Dgm;

    using opts::Option;
    using opts::PosOption;

    opts::Options ops;

    size_t n_pixels = 50;
    int n_dgm_points = 1000000;
    int n_threads = 1;

    ops >> Option('p', "pixels", n_pixels, "pixels along x/y axis")
        >> Option('t', "threads", n_threads, "# threads")
        >> Option('d', "diagram", n_dgm_points, "# diagram points");

    if (!ops.parse(argc, argv)) {
        std::cout << "Usage: " << argv[0] << " [options]\n\n";
        std::cout << ops << std::endl;
        return 1;
    }

    oineus::ImageResolution resolution{n_pixels, n_pixels};

    Dgm dgm;

    int seed = 1;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<Real> dis(0.0, 10.0);
    for (int i = 0; i < n_dgm_points; ++i) {
        Real x = dis(gen);
        Real y = x + dis(gen);
        dgm.emplace_back(x, y);
    }

    std::cerr << "Random diagram generated" << std::endl;

    Real sigma = 1.0;
    Vectorizer vectorizer(sigma, resolution);
    vectorizer.set_verbose(true);
    vectorizer.set_n_threads(n_threads);

    std::cerr << "With erf:\n";
    auto image = vectorizer.persistence_image_dirac_unstable(dgm);

    vectorizer.set_n_threads(1);

    auto image_1 = vectorizer.persistence_image_dirac_unstable_serial(dgm);
    Real diff = 0;
    Real sum = std::accumulate(image.begin(), image.end(), Real(0));
    Real sum_1 = std::accumulate(image.begin(), image.end(), Real(0));
    for(size_t i = 0; i < image_1.size(); ++i) {
        diff += abs(image_1[i] - image[i]);
        if (abs(image_1[i] - image[i]) > 0)
            std::cerr << "i = " << i << ", " << image[i] << " != " << image_1[i] << std::endl;
    }
    std::cerr << "results equal: " << (image == image_1) <<  ", diff = " << diff << ", sum = " << sum << ", sum_1 = " << sum_1 << std::endl;

    std::cerr << "Dirac weighting - without erf:\n";
    image = vectorizer.persistence_image_unstable(dgm);

    return 0;
}