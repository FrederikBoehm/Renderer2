#include "material/material.hpp"
#include <assimp/material.h>

namespace rt {
  CMaterial::CMaterial() :
    m_Le(glm::vec3(0.0f)),
    m_orenNayarBRDF(),
    m_microfacetBRDF() {
  }

  CMaterial::CMaterial(const glm::vec3& le) :
    m_Le(le),
    m_orenNayarBRDF(),
    m_microfacetBRDF() {

  }

  CMaterial::CMaterial(const COrenNayarBRDF& diffuse, const CMicrofacetBRDF& glossy) :
    m_Le(glm::vec3(0.0f)),
    m_orenNayarBRDF(diffuse),
    m_microfacetBRDF(glossy) {

  }

  float CMaterial::roughnessFromExponent(float exponent) const {
    return powf(2.f / (exponent + 2.f), 0.25f);
  }
}