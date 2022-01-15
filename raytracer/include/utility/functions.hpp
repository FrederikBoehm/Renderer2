#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

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

   template <glm::length_t L, typename T> DH_CALLABLE bool any(glm::vec<L, T>& vec, T v) {
     for (glm::length_t i = 0; i < L; ++i) {
       if (vec[i] == v) {
         return true;
       }
     }
     return false;
   }

   DH_CALLABLE inline bool sameHemisphere(const glm::vec3& w, const glm::vec3& wp) {
     return w.z * wp.z > 0.f;
   }

   DH_CALLABLE inline float interpolate(float t, float v1, float v2) {
     return (1.f - t) * v1 + t * v2;
   }

   template <typename T> DH_CALLABLE inline bool insideExclusive(const glm::vec<3, T>& p, const glm::vec<3, T>& lowerBound, const glm::vec<3, T>& upperBound) {
     return (p.x >= lowerBound.x && p.x < upperBound.x &&
             p.y >= lowerBound.y && p.y < upperBound.y &&
             p.z >= lowerBound.z && p.z < upperBound.z);
   }

   template <typename T> DH_CALLABLE inline bool inside(const glm::vec<3, T>& p, const glm::vec<3, T>& lowerBound, const glm::vec<3, T>& upperBound) {
     return (p.x >= lowerBound.x && p.x <= upperBound.x &&
       p.y >= lowerBound.y && p.y <= upperBound.y &&
       p.z >= lowerBound.z && p.z <= upperBound.z);
   }

   DH_CALLABLE inline uint32_t floatsToBits(float f) {
     uint32_t ui;
     memcpy(&ui, &f, sizeof(float));
     return ui;
   }

   DH_CALLABLE inline float bitsToFloat(uint32_t ui) {
     float f;
     memcpy(&f, &ui, sizeof(uint32_t));
     return f;
   }

   DH_CALLABLE inline float nextFloatUp(float v) {
     if (glm::isinf(v) && v > 0.f) {
       return v;
     }
     if (v == -0.f) {
       return 0.f;
     }
     uint32_t ui = floatsToBits(v);
     if (v >= 0) {
       ++ui;
     }
     else {
       --ui;
     }
     return bitsToFloat(ui);
   }

   DH_CALLABLE inline float nextFloatDown(float v) {
     if (glm::isinf(v) && v < 0.f) {
       return v;
     }
     if (v == 0.f) {
       return -0.f;
     }
     uint32_t ui = floatsToBits(v);
     if (v > 0) {
       --ui;
     }
     else {
       ++ui;
     }
     return bitsToFloat(ui);
   }

   DH_CALLABLE inline glm::mat4 getRotation(const glm::vec3& normal) {
     float cos = glm::dot(normal, glm::vec3(0.0f, 1.0f, 0.0f));
     if (cos == 1.0f) {
       return glm::mat4(1.0f);
     }
     else if (cos == -1.0f) {
       return glm::mat4(glm::mat3(-1.0f));
     }
     else {
       float angle = glm::acos(glm::dot(normal, glm::vec3(0.0f, 1.0f, 0.0f)));
       glm::vec3 rotationAxis = glm::normalize(glm::cross(normal, glm::vec3(0.0f, 1.0f, 0.0f)));
       return glm::rotate(glm::mat4(1.0f), angle, rotationAxis);
     }
   }
}

#endif // !FUNCTIONS_HPP
