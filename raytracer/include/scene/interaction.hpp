#ifndef SURFACE_INTERACTION_HXX
#define SURFACE_INTERACTION_HXX

#include "intersect/hit_information.hpp"
#include "material/material.hpp"
#include "medium/medium.hpp"

namespace rt {
  class CDeviceSceneobject;
  struct SInteraction {
    SHitInformation hitInformation;
    const CDeviceSceneobject* object = nullptr;
    CMaterial* material = nullptr;
    const CMedium* medium = nullptr;
  };

  //struct SSurfaceInteraction : public SInteraction {
  //  CMaterial* material;
  //};

  //struct SMediumInteraction : public SInteraction {

  //};
}

#endif // !SURFACE_INTERACTION_HXX
