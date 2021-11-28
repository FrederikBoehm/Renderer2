#ifndef PHASE_FUNCTION_HPP
#define PHASE_FUNCTION_HPP

#include "utility/qualifiers.hpp"
#include <glm/glm.hpp>
namespace rt {
  enum EPhaseFunction {
    HENYEY_GREENSTEIN,
    SGGX
  };

  class CSampler;

  class CPhaseFunction {
  protected:
    DH_CALLABLE CPhaseFunction(const EPhaseFunction type);

  public:
    DH_CALLABLE virtual ~CPhaseFunction() = default;
    DH_CALLABLE float p(const glm::vec3& wo, const glm::vec3& wi, const glm::vec3& n, CSampler& sampler) const;
    D_CALLABLE float sampleP(const glm::vec3& wo, glm::vec3* wi, const glm::vec3& n, CSampler& sampler) const;

    DH_CALLABLE EPhaseFunction type() const;

  private:
    EPhaseFunction m_type;
  };

  inline EPhaseFunction CPhaseFunction::type() const {
    return m_type;
  }

  
}

#endif
