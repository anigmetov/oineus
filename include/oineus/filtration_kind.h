#pragma once

#include <iosfwd>
#include <sstream>
#include <string>

namespace oineus {

    // Coarse classification of how the filtration was constructed.
    // Used by oineus.diff to pick reduction defaults (homology vs
    // cohomology, diagram-extraction side) and by oineus.vis to pick
    // points-vs-scalar-field rendering. `User` is the safe default for
    // hand-built filtrations.
    enum class FiltrationKind {
        User,
        Vr,
        Alpha,
        WeakAlpha,
        CechDelaunay,
        Freudenthal,
        Cubical,
        MinFil,
        MappingCylinder,
    };

    inline const char* filtration_kind_name(FiltrationKind k)
    {
        switch (k) {
            case FiltrationKind::User:            return "user";
            case FiltrationKind::Vr:              return "vr";
            case FiltrationKind::Alpha:           return "alpha";
            case FiltrationKind::WeakAlpha:       return "weak_alpha";
            case FiltrationKind::CechDelaunay:    return "cech_delaunay";
            case FiltrationKind::Freudenthal:     return "freudenthal";
            case FiltrationKind::Cubical:         return "cubical";
            case FiltrationKind::MinFil:          return "min_filtration";
            case FiltrationKind::MappingCylinder: return "mapping_cylinder";
        }
        return "unknown";
    }

    inline std::ostream& operator<<(std::ostream& out, FiltrationKind k)
    {
        return out << filtration_kind_name(k);
    }

    inline std::string to_string(FiltrationKind k)
    {
        return filtration_kind_name(k);
    }

} // namespace oineus
