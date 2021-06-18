#ifndef SURFACE_INTERACTION_HXX
#define SURFACE_INTERACTION_HXX

#include "intersect/hit_information.hpp"
#include "material/material.hpp"

namespace rt {
  struct SSurfaceInteraction {
    SHitInformation hitInformation;
    CMaterial material;
  };
}

#endif // !SURFACE_INTERACTION_HXX
