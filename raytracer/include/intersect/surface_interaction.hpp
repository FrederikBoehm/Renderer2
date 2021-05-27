#ifndef SURFACE_INTERACTION_HXX
#define SURFACE_INTERACTION_HXX

#include <cstdint>
#include <glm/glm.hpp>

namespace rt {
  struct SurfaceInteraction {
    bool hit;
    glm::vec3 pos; // World pos
    uint16_t objectId;
  };
}
#endif