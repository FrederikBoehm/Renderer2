#define _USE_MATH_DEFINES
#include <math.h>
#include "medium/henyey_greenstein_phase_function.hpp"
#include <algorithm>
#include "utility/functions.hpp"
#include <glm/glm.hpp>

namespace rt {
  DH_CALLABLE inline float phaseHG(float cosTheta, float g) {
    float denom = 1 + g * g + 2 * g * cosTheta;
    return (1 - g * g) / (4 * M_PI * denom * glm::sqrt(denom));
  }

  CHenyeyGreensteinPhaseFunction::CHenyeyGreensteinPhaseFunction(float g) : m_g(g) {

  }

  float CHenyeyGreensteinPhaseFunction::p(const glm::vec3& wo, const glm::vec3& wi) const {
    float cosTheta = glm::dot(wo, wi);
    return phaseHG(cosTheta, m_g);
  }

  float CHenyeyGreensteinPhaseFunction::sampleP(const glm::vec3& wo, glm::vec3* wi, const glm::vec2& u) const {
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

