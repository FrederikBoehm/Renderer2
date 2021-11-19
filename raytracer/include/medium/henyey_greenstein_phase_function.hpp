#ifndef HENYEY_GREENSTEIN_PHASE_FUNCTION_HPP
#define HENYEY_GREENSTEIN_PHASE_FUNCTION_HPP

#include "glm/glm.hpp"
#include "utility/qualifiers.hpp"
#include "phase_function.hpp"

namespace rt {
  class CHenyeyGreensteinPhaseFunction : public CPhaseFunction {
  public:
    DH_CALLABLE CHenyeyGreensteinPhaseFunction(float g);

    DH_CALLABLE float p(const glm::vec3& wo, const glm::vec3& wi) const;
    DH_CALLABLE float sampleP(const glm::vec3& wo, glm::vec3* wi, const glm::vec2& u) const;

  private:
    float m_g;
  };
}

#endif