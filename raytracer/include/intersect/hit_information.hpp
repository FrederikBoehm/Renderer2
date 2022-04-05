#ifndef HIT_INFORMATION_HXX
#define HIT_INFORMATION_HXX

#include <cstdint>
#include <glm/glm.hpp>

namespace rt {
  struct SHitInformation {
    bool hit = false;
    glm::vec3 pos; // World pos
    glm::vec3 normal; // World space normal (including normalmapping)
    glm::vec3 normalG; // World space geometry normal
    glm::mat3 sggxS; // For medium interaction: Matrix S for SGGX Distribution
    glm::vec2 tc;
    float ior;
    float t;
  };
}
#endif