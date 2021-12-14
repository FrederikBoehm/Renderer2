#ifndef HENYEY_GREENSTEIN_PHASE_FUNCTION_HPP
#define HENYEY_GREENSTEIN_PHASE_FUNCTION_HPP
#define _USE_MATH_DEFINES
#include <math.h>
#include "glm/glm.hpp"
#include "utility/qualifiers.hpp"
#include "phase_function.hpp"
#include "utility/functions.hpp"

namespace rt {
  class CHenyeyGreensteinPhaseFunction : public CPhaseFunction {
  public:
    H_CALLABLE CHenyeyGreensteinPhaseFunction(float g);

    DH_CALLABLE float p(const glm::vec3& wo, const glm::vec3& wi) const;
    DH_CALLABLE float sampleP(const glm::vec3& wo, glm::vec3* wi, const glm::vec2& u) const;

  private:
    float m_g;
  };

  DH_CALLABLE inline float phaseHG(float cosTheta, float g) {
    float denom = 1 + g * g + 2 * g * cosTheta;
    return (1 - g * g) / (4 * M_PI * denom * glm::sqrt(denom));
  }

  inline float CHenyeyGreensteinPhaseFunction::p(const glm::vec3& wo, const glm::vec3& wi) const {
    float cosTheta = glm::dot(wo, wi);
    return phaseHG(cosTheta, m_g);
  }

  inline float CHenyeyGreensteinPhaseFunction::sampleP(const glm::vec3& wo, glm::vec3* wi, const glm::vec2& u) const {
    float cosTheta;
    if (glm::abs(m_g) < 1e-3f) {
      cosTheta = 1.f - 2.f * u.x;
    }
    else {
      float sqrTerm = (1.f - m_g * m_g) / (1.f - m_g + 2.f * m_g * u.x);
      cosTheta = (1.f + m_g * m_g - sqrTerm * sqrTerm) / (2.f * m_g);
    }

    float sinTheta = glm::sqrt(glm::max(0.f, 1.f - cosTheta * cosTheta));
    float phi = 2.f * M_PI * u.y;
    glm::vec3 v1;
    glm::vec3 v2;

    coordinateSystem(wo, &v1, &v2);
    *wi = sphericalDirection(sinTheta, cosTheta, phi, v1, v2, -wo);
    return phaseHG(-cosTheta, m_g);
  }
}

#endif