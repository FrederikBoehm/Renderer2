#include <glm/glm.hpp>
#include "qualifiers.hpp"
namespace rt {
  DH_CALLABLE inline float luma(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
  }

  DH_CALLABLE inline float luma(const glm::vec3& clr) {
    return luma(clr.r, clr.g, clr.b);
  }
}