#ifndef SGGX_PHASE_FUNCTION_HPP
#define SGGX_PHASE_FUNCTION_HPP

#include "utility/qualifiers.hpp"
#include <glm/glm.hpp>
#include "medium/phase_function.hpp"
#include "integrators/objects.hpp"
#include "sampling/sampler.hpp"
#include "material/fresnel.hpp"

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
    DH_CALLABLE CSGGXMicroflakeDistribution(const glm::mat3& S, const glm::vec3& normal);

    DH_CALLABLE float D(const glm::vec3& w) const;
    DH_CALLABLE float sigma(const glm::vec3& w) const;
    DH_CALLABLE glm::vec3 sampleVNDF(const glm::vec3& w_i, const glm::vec2& U) const;
  
    DH_CALLABLE static glm::mat3 buildS(const glm::vec3& n, float sigma);

  private:
    const glm::mat3 m_S;
    const glm::mat3 m_invS;
    const glm::vec3 m_normal;

    DH_CALLABLE glm::mat3 projectS(const glm::mat3& S, const glm::vec3& w_k, const glm::vec3& w_j, const glm::vec3& w_i) const;
  };

  class CSGGXSpecularPhaseFunction {
  public:
    DH_CALLABLE CSGGXSpecularPhaseFunction(const CSGGXMicroflakeDistribution& distribution);
    
    DH_CALLABLE float p(const glm::vec3& w_o, const glm::vec3& w_i) const;
    D_CALLABLE float sampleP(const glm::vec3& wo, glm::vec3* wi, CSampler& sampler) const;
  
  private:
    const CSGGXMicroflakeDistribution& m_distribution;
  };

  class CSGGXDiffusePhaseFunction {
  public:
    DH_CALLABLE CSGGXDiffusePhaseFunction(const CSGGXMicroflakeDistribution& distribution);

    DH_CALLABLE float p(const glm::vec3& w_o, const glm::vec3& w_i, CSampler& sampler) const;
    D_CALLABLE float sampleP(const glm::vec3& wo, glm::vec3* wi, CSampler& sampler) const;

  private:
    const CSGGXMicroflakeDistribution& m_distribution;
    
    DH_CALLABLE float p(const glm::vec3& w_o, const glm::vec3& w_i, const glm::vec3& normal) const;
  };

  class CSGGXPhaseFunction {
  public:
    DH_CALLABLE CSGGXPhaseFunction(const glm::mat3& S, const glm::vec3& normal, float ior);

    DH_CALLABLE float p(const glm::vec3& w_o, const glm::vec3& w_i, CSampler& sampler) const;
    D_CALLABLE float sampleP(const glm::vec3& wo, glm::vec3* wi, CSampler& sampler) const;
    DH_CALLABLE glm::vec3 sampleVNDF(const glm::vec3& w_i, const glm::vec2& U) const;

  private:
    const CSGGXMicroflakeDistribution m_distribution;
    const CSGGXDiffusePhaseFunction m_diffuse;
    const CSGGXSpecularPhaseFunction m_specular;
    const glm::vec3 m_normal;
    const float m_ior;
  };

  class CSGGXPhaseFunctionPlaceholder : public CPhaseFunction {
  public:
    H_CALLABLE CSGGXPhaseFunctionPlaceholder();
  };

  // CSGGXMicroflakeDistribution

  inline CSGGXMicroflakeDistribution::CSGGXMicroflakeDistribution(const glm::mat3& S, const glm::vec3& normal) :
    m_S(S),
    m_invS(glm::inverse(S)),
    m_normal(normal) {
  }

  inline glm::mat3 CSGGXMicroflakeDistribution::buildS(const glm::vec3& n, float sigma) {
    float xx = n.x * n.x;
    float xy = n.x * n.y;
    float xz = n.x * n.z;
    float yy = n.y * n.y;
    float yz = n.y * n.z;
    float zz = n.z * n.z;
    glm::mat3 m1(xx, xy, xz, xy, yy, yz, xz, yz, zz);
    glm::mat3 m2(yy + zz, -xy, -xz, -xy, xx + zz, -yz, -xz, -yz, xx + yy);
    return m1 + sigma * m2;
  }

  inline float CSGGXMicroflakeDistribution::D(const glm::vec3& w) const {
    float v = glm::dot(w, m_invS * w);
    return 1.f / (M_PI * glm::sqrt(glm::abs(glm::determinant(m_S))) * v * v);
  }

  inline float CSGGXMicroflakeDistribution::sigma(const glm::vec3& w) const {
    return glm::sqrt(glm::max(glm::dot(w, m_S * w), 0.f));
  }

  inline glm::vec3 CSGGXMicroflakeDistribution::sampleVNDF(const glm::vec3& w_i, const glm::vec2& U) const {
    const CCoordinateFrame frame = CCoordinateFrame::fromNormal2(m_normal);
    const glm::vec3& w_k = frame.B();
    const glm::vec3& w_j = frame.T();

    const glm::mat3 S_kji = projectS(m_S, w_k, w_j, m_normal);

    const float S_kk = S_kji[0][0];
    const float S_kj = S_kji[0][1];
    const float S_ki = S_kji[0][2];
    const float S_jj = S_kji[1][1];
    const float S_ji = S_kji[1][2];
    const float S_ii = S_kji[2][2];

    const float temp = glm::sqrt(S_jj*S_ii - S_ji * S_ji);

    const glm::vec3 M_k(glm::sqrt(glm::abs(glm::determinant(S_kji))) / temp, 0.f, 0.f);
    const float scaling = 1.f / glm::sqrt(S_ii);
    const glm::vec3 M_j = glm::vec3(-(S_ki*S_ji - S_kj * S_ii) / temp, temp, 0.f) * scaling;
    const glm::vec3 M_i = glm::vec3(S_kj, S_ji, S_ii) * scaling;

    const float sqrt_Ux = glm::sqrt(U.x);
    const float u = sqrt_Ux * glm::cos(2.f * M_PI * U.y);
    const float v = sqrt_Ux * glm::sin(2.f * M_PI * U.y);
    const float w = glm::sqrt(1.f - u * u - v * v);

    glm::vec3 w_m = u * M_k + v * M_j + w * M_i;
    w_m = glm::normalize(w_m);
    return glm::normalize(glm::mat3(w_k, w_j, m_normal) * w_m);
  }

  inline glm::mat3 CSGGXMicroflakeDistribution::projectS(const glm::mat3& S, const glm::vec3& w_k, const glm::vec3& w_j, const glm::vec3& w_i) const {
    float v_kk = glm::dot(w_k, S * w_k);
    float v_kj = glm::dot(w_k, S * w_j);
    float v_ki = glm::dot(w_k, S * w_i);
    float v_jj = glm::dot(w_j, S * w_j);
    float v_ji = glm::dot(w_j, S * w_i);
    float v_ii = glm::dot(w_i, S * w_i);

    return glm::mat3(v_kk, v_kj, v_ki,
      v_kj, v_jj, v_ji,
      v_ki, v_ji, v_ii);
  }


  // CSGGXSpecularPhaseFunction

  inline CSGGXSpecularPhaseFunction::CSGGXSpecularPhaseFunction(const CSGGXMicroflakeDistribution& distribution):
    m_distribution(distribution) {

  }

  inline float CSGGXSpecularPhaseFunction::p(const glm::vec3& w_o, const glm::vec3& w_i) const {
    const glm::vec3 w_h = glm::normalize(w_i + w_o);
    return m_distribution.D(w_h) / (4.f * m_distribution.sigma(w_o)); // w_o because of different convention for incoming and outgoing directions
  }

  inline float CSGGXSpecularPhaseFunction::sampleP(const glm::vec3& wo, glm::vec3* wi, CSampler& sampler) const {
    glm::vec3 w_m = m_distribution.sampleVNDF(wo, glm::vec2(sampler.uniformSample01(), sampler.uniformSample01()));
    *wi = glm::reflect(-wo, w_m);
    return p(wo, *wi);
  }

  // CSGGXDiffusePhaseFunction

  inline CSGGXDiffusePhaseFunction::CSGGXDiffusePhaseFunction(const CSGGXMicroflakeDistribution& distribution):
    m_distribution(distribution) {

  }

  inline float CSGGXDiffusePhaseFunction::p(const glm::vec3& w_o, const glm::vec3& w_i, CSampler& sampler) const {
    glm::vec3 w_m = m_distribution.sampleVNDF(w_o, glm::vec2(sampler.uniformSample01(), sampler.uniformSample01()));
    return glm::max(glm::dot(w_i, w_m), 0.f) / M_PI;
  }

  inline float CSGGXDiffusePhaseFunction::p(const glm::vec3& w_o, const glm::vec3& w_i, const glm::vec3& normal) const {
    return glm::max(glm::dot(w_i, normal), 0.f) / M_PI;
  }

  inline float CSGGXDiffusePhaseFunction::sampleP(const glm::vec3& wo, glm::vec3* wi, CSampler& sampler) const {
    glm::vec3 w_n = m_distribution.sampleVNDF(wo, glm::vec2(sampler.uniformSample01(), sampler.uniformSample01()));
    CCoordinateFrame frame = CCoordinateFrame::fromNormal(w_n);
    glm::vec3 w_m = sampler.uniformSampleHemisphere();
    glm::vec3 w_m_world = frame.tangentToWorld() * glm::vec4(w_m, 0.f);
    *wi = glm::normalize(w_m_world); // TODO: Check coordinate space
    return p(wo, *wi, w_n);
  }

  // CSGGXPhaseFunction

  inline CSGGXPhaseFunction::CSGGXPhaseFunction(const glm::mat3& S, const glm::vec3& normal, float ior) :
    m_distribution(S, normal),
    m_diffuse(m_distribution),
    m_specular(m_distribution),
    m_normal(normal),
    m_ior(ior) {
  }

  inline float CSGGXPhaseFunction::p(const glm::vec3& w_o, const glm::vec3& w_i, CSampler& sampler) const {
    CFresnel fresnel(1.f, m_ior);
    float weight = fresnel.evaluate(glm::abs(glm::dot(w_o, m_normal)));
    return m_diffuse.p(w_o, w_i, sampler) * (1.f - weight) + m_specular.p(w_o, w_i) * weight;
  }

  inline float CSGGXPhaseFunction::sampleP(const glm::vec3& wo, glm::vec3* wi, CSampler& sampler) const {
    CFresnel fresnel(1.f, m_ior);
    float p = fresnel.evaluate(glm::abs(glm::dot(wo, m_normal)));
    if (sampler.uniformSample01() < p) {
      return m_specular.sampleP(wo, wi, sampler);
    }
    else {
      return m_diffuse.sampleP(wo, wi, sampler);
    }
  }

  inline glm::vec3 CSGGXPhaseFunction::sampleVNDF(const glm::vec3& w_i, const glm::vec2& U) const {
    return m_distribution.sampleVNDF(w_i, U);
  }
}

#endif