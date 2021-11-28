#ifndef SGGX_PHASE_FUNCTION_HPP
#define SGGX_PHASE_FUNCTION_HPP

#include "utility/qualifiers.hpp"
#include <glm/glm.hpp>
#include "medium/phase_function.hpp"

namespace rt {
  struct SSGGXDistributionParameters {
    float m_Sxx;
    float m_Syy;
    float m_Szz;
    float m_Sxy;
    float m_Sxz;
    float m_Syz;
  };

  class CCoordinateFrame;
  class CSampler;

  class CSGGXMicroflakeDistribution {
  public:
    H_CALLABLE CSGGXMicroflakeDistribution(const SSGGXDistributionParameters& parameters);
    H_CALLABLE CSGGXMicroflakeDistribution(float sigma);

    DH_CALLABLE float D(const glm::vec3& w) const;
    DH_CALLABLE float D(const glm::vec3& w, const glm::mat3& S) const;
    DH_CALLABLE float sigma(const glm::vec3& w) const;
    DH_CALLABLE float sigma(const glm::vec3& w, const glm::mat3& S) const;
    DH_CALLABLE glm::vec3 sampleVNDF(const glm::vec3& w_i, const glm::vec2& U) const;
    DH_CALLABLE glm::vec3 sampleVNDF(const glm::vec3& w_i, const glm::vec2& U, const glm::mat3& S) const;
  
    DH_CALLABLE glm::mat3 buildS(const glm::vec3& n) const;

  private:
    const glm::mat3 m_S;
    const glm::mat3 m_invS;
    const float m_sigma;

    H_CALLABLE static glm::mat3 getS(const SSGGXDistributionParameters& parameters);
    DH_CALLABLE glm::mat3 projectS(const glm::mat3& S, const glm::vec3& w_k, const glm::vec3& w_j, const glm::vec3& w_i) const;
  };

  class CSGGXSpecularPhaseFunction {
  public:
    H_CALLABLE CSGGXSpecularPhaseFunction(const SSGGXDistributionParameters& distributionsParameters);
    H_CALLABLE CSGGXSpecularPhaseFunction(float sigma);
    
    DH_CALLABLE float p(const glm::vec3& w_o, const glm::vec3& w_i, const glm::vec3& n) const;
    D_CALLABLE float sampleP(const glm::vec3& wo, glm::vec3* wi, const glm::vec3& n, CSampler& sampler) const;
  
  private:
    const CSGGXMicroflakeDistribution m_distribution;
  };

  class CSGGXDiffusePhaseFunction {
  public:
    H_CALLABLE CSGGXDiffusePhaseFunction(const SSGGXDistributionParameters& distributionsParameters);
    H_CALLABLE CSGGXDiffusePhaseFunction(float sigma);

    DH_CALLABLE float p(const glm::vec3& w_o, const glm::vec3& w_i, const glm::vec3& normal, CSampler& sampler) const;
    D_CALLABLE float sampleP(const glm::vec3& wo, glm::vec3* wi, const glm::vec3& n, CSampler& sampler) const;

  private:
    const CSGGXMicroflakeDistribution m_distribution;
    
    DH_CALLABLE float p(const glm::vec3& w_o, const glm::vec3& w_i, const glm::vec3& normal) const;
  };

  class CSGGXPhaseFunction : public CPhaseFunction {
  public:
    H_CALLABLE CSGGXPhaseFunction(const SSGGXDistributionParameters& diffuseParameters, const SSGGXDistributionParameters& specularParameters);
    H_CALLABLE CSGGXPhaseFunction(float diffuseSigma, float specularSigma);

    DH_CALLABLE float p(const glm::vec3& w_o, const glm::vec3& w_i, const glm::vec3& n, CSampler& sampler) const;
    D_CALLABLE float sampleP(const glm::vec3& wo, glm::vec3* wi, const glm::vec3& n, CSampler& sampler) const;

  private:
    const CSGGXDiffusePhaseFunction m_diffuse;
    const CSGGXSpecularPhaseFunction m_specular;
  };
}

#endif