#ifndef SURFACE_INTERACTION_HXX
#define SURFACE_INTERACTION_HXX

#include "intersect/hit_information.hpp"
#include "material/material.hpp"

namespace rt {
  class CDeviceSceneobject;
  struct SSurfaceInteraction {
    SHitInformation hitInformation;
    CMaterial material;
    CDeviceSceneobject* object;
  };
}

#endif // !SURFACE_INTERACTION_HXX
