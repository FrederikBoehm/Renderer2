#ifndef SURFACE_INTERACTION_HXX
#define SURFACE_INTERACTION_HXX

#include "intersect/hit_information.hpp"

namespace rt {
  class CDeviceSceneobject;
  class CMaterial;
  class CMediumInstance;
  struct SInteraction {
    SHitInformation hitInformation;
    const CDeviceSceneobject* object = nullptr;
    CMaterial* material = nullptr;
    const CMediumInstance* medium = nullptr;
  };

  //struct SSurfaceInteraction : public SInteraction {
  //  CMaterial* material;
  //};

  //struct SMediumInteraction : public SInteraction {

  //};
}

#endif // !SURFACE_INTERACTION_HXX
