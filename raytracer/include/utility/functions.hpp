#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP
#include <glm/glm.hpp>

#include "qualifiers.hpp"

namespace rt {
   template <typename Predicate> DH_CALLABLE int findInterval(int size, const Predicate& pred) {
    int first = 0;
    int len = size;
    while (len > 0) {
      int half = len >> 1;
      int middle = first + half;

      if (pred(middle)) {
        first = middle + 1;
        len -= half + 1;
      }
      else {
        len = half;
      }
    }

    return glm::clamp(first - 1, 0, size - 2);
  }

   DH_CALLABLE inline void coordinateSystem(const glm::vec3& v1, glm::vec3* v2, glm::vec3* v3) {
     if (glm::abs(v1.x) > glm::abs(v1.y)) {
       *v2 = glm::vec3(-v1.z, 0.f, v1.x) / glm::sqrt(v1.x * v1.x + v1.z * v1.z);
     }
     else {
       *v2 = glm::vec3(0.f, v1.z, -v1.y) / glm::sqrt(v1.y * v1.y + v1.z * v1.z);
     }
     *v3 = glm::cross(v1, *v2);
   }

   DH_CALLABLE inline glm::vec3 sphericalDirection(float sinTheta, float cosTheta, float phi) {
     return glm::vec3(sinTheta * glm::cos(phi),
                      sinTheta * glm::sin(phi),
                      cosTheta); // TODO: Check if that is the right coordinate system
   }

   DH_CALLABLE inline glm::vec3 sphericalDirection(float sinTheta, float cosTheta, float phi, const glm::vec3& x, const glm::vec3& y, const glm::vec3& z) {
     return sinTheta * glm::cos(phi) * x + sinTheta * glm::sin(phi) * y + cosTheta * z; // TODO: Check if that is the right coordinate system
   }
}

#endif // !FUNCTIONS_HPP
