#include <string>
#include <vector>
#include <iostream>

#include <opts/opts.h>

int main(int argc, char** argv)
{
    using opts::Option;
    using opts::PosOption;

    opts::Options ops;

    std::string                     name    = "Pepper";
    std::string                     prefix  = ".";
    unsigned int                    age     = 7;
    double                          area    = 1.0;
    std::vector<std::string>        coordinates;
    bool negate, help;
    ops
        >> Option(      "name",     name,                           "name of the person")
        >> Option('p',  "path",     prefix,         "PREFIX",       "path prefix")
        >> Option('a',  "age",      age,                            "age of the person")
        >> Option(      "area",     area,                           "some area")
        >> Option('c',  "coord",    coordinates,    "X Y ...",      "coordinates")
        >> Option('n',  "negate",   negate,                         "negate the function")
        >> Option('h',  "help",     help,                           "show help message")
    ;

    // NB: PosOptions must be read last
    std::string infilename, outfilename;
    int repeat;
    if ( !ops.parse(argc, argv) || help ||
        !(ops >> PosOption(infilename) >> PosOption(repeat) >> PosOption(outfilename)))
    {
        std::cout << "Usage: " << argv[0] << " [options] INFILE REPEAT OUTFILE\n\n";
        std::cout << "Sample options program\n\n";
        std::cout << ops << std::endl;
        return 1;
    }

    std::cout << "Infilename:  " << infilename  << std::endl;
    std::cout << "Repeat:      " << repeat      << std::endl;
    std::cout << "Outfilename: " << outfilename << std::endl;
    std::cout << "Name:        " << name        << std::endl;
    std::cout << "Prefix:      " << prefix      << std::endl;
    std::cout << "Area:        " << area        << std::endl;
    std::cout << "Age:         " << age         << std::endl;
    std::cout << "Negate:      " << negate      << std::endl;
    std::cout << "Coorindates: " << std::endl;
    for (unsigned i = 0; i < coordinates.size(); ++i)
        std::cout << "  " << coordinates[i] << std::endl;
}
