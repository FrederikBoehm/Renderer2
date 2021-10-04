#ifndef BRDF_FUNCTIONS_HPP
#define BRDF_FUNCTIONS_HPP

#include "utility/qualifiers.hpp"

namespace rt {
  DH_CALLABLE inline float cosTheta(const glm::vec3& w) {
    return w.z;
  }

  DH_CALLABLE inline float cos2Theta(const glm::vec3& w) {
    return w.z * w.z;
  }

  DH_CALLABLE inline float absCosTheta(const glm::vec3& w) {
    return glm::abs(w.z);
  }

  DH_CALLABLE inline float sin2Theta(const glm::vec3& w) {
    return glm::max(0.0f, 1.0f - cos2Theta(w));
  }

  DH_CALLABLE inline float sinTheta(const glm::vec3& w) {
    return glm::sqrt(sin2Theta(w));
  }

  DH_CALLABLE inline float tanTheta(const glm::vec3& w) {
    return sinTheta(w) / cosTheta(w);
  }

  DH_CALLABLE inline float tan2Theta(const glm::vec3& w) {
    return sin2Theta(w) / cos2Theta(w);
  }

  DH_CALLABLE inline float cosPhi(const glm::vec3& w) {
    float sinThetaEvaluated = sinTheta(w);
    return (sinThetaEvaluated == 0.0f) ? 1.0f : glm::clamp(w.x / sinThetaEvaluated, -1.0f, 1.0f);
  }

  DH_CALLABLE inline float sinPhi(const glm::vec3& w) {
    float sinThetaEvaluated = sinTheta(w);
    return (sinThetaEvaluated == 0.0f) ? 0.0f : glm::clamp(w.y / sinThetaEvaluated, -1.0f, 1.0f);
  }

  DH_CALLABLE inline float cos2Phi(const glm::vec3& w) {
    float cosPhiEvaluated = cosPhi(w);
    return cosPhiEvaluated * cosPhiEvaluated;
  }

  DH_CALLABLE inline float sin2Phi(const glm::vec3& w) {
    float sinPhiEvaluated = sinPhi(w);
    return sinPhiEvaluated * sinPhiEvaluated;
  }
}

#endif