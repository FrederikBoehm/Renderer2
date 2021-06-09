#ifndef SURFACE_INTERACTION_HXX
#define SURFACE_INTERACTION_HXX

#include "intersect/hit_information.hpp"

namespace rt {
  struct SSurfaceInteraction {
    SHitInformation hitInformation;
    glm::vec3 surfaceAlbedo;
  };
}

#endif // !SURFACE_INTERACTION_HXX
