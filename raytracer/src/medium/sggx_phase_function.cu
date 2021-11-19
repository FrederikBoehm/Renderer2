#define _USE_MATH_DEFINES
#include <math.h>

#include "medium/sggx_phase_function.hpp"
#include "integrators/objects.hpp"
#include "sampling/sampler.hpp"
#include <stdio.h>

namespace rt {
  // CSGGXMicroflakeDistribution

  CSGGXMicroflakeDistribution::CSGGXMicroflakeDistribution(const SSGGXDistributionParameters& parameters):
    m_S(getS(parameters)),
    m_invS(glm::inverse(m_S)) {

  }

  glm::mat3 CSGGXMicroflakeDistribution::getS(const SSGGXDistributionParameters& parameters) {
    glm::mat3 S(parameters.m_Sxx, parameters.m_Sxy, parameters.m_Sxz,
                parameters.m_Sxy, parameters.m_Syy, parameters.m_Syz,
                parameters.m_Sxz, parameters.m_Syz, parameters.m_Szz);
    return S;
  }

  float CSGGXMicroflakeDistribution::D(const glm::vec3& w) const {
    float v = glm::dot(w, m_invS * w);
    return 1.f / (M_PI * glm::sqrt(glm::abs(glm::determinant(m_S))) * v * v);
  }

  float CSGGXMicroflakeDistribution::sigma(const glm::vec3& w) const {
    return glm::sqrt(glm::max(glm::dot(w, m_S * w), 0.f));
  }

  glm::vec3 CSGGXMicroflakeDistribution::sampleVNDF(const glm::vec3& w_i, const glm::vec2& U) const {
    const CCoordinateFrame frame = CCoordinateFrame::fromNormal(w_i);
    const glm::vec3& w_k = frame.B();
    const glm::vec3& w_j = frame.T();

    const glm::mat3 S_kji = projectS(m_S, w_k, w_j, w_i);
    
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
    return glm::normalize(glm::mat3(w_k, w_j, w_i) * w_m);
  }

  glm::mat3 CSGGXMicroflakeDistribution::projectS(const glm::mat3& S, const glm::vec3& w_k, const glm::vec3& w_j, const glm::vec3& w_i) const {
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

  CSGGXSpecularPhaseFunction::CSGGXSpecularPhaseFunction(const SSGGXDistributionParameters& distributionsParameters):
    m_distribution(distributionsParameters) {

  }

  float CSGGXSpecularPhaseFunction::p(const glm::vec3& w_o, const glm::vec3& w_i) const {
    const glm::vec3 w_h = glm::normalize(w_i + w_o);
    return m_distribution.D(w_h) / (4.f * m_distribution.sigma(w_i));
  }

  float CSGGXSpecularPhaseFunction::sampleP(const glm::vec3& wo, glm::vec3* wi, CSampler& sampler) const {
    glm::vec3 w_m = m_distribution.sampleVNDF(wo, glm::vec2(sampler.uniformSample01(), sampler.uniformSample01()));
    *wi = glm::reflect(-wo, w_m);
    return p(*wi, wo);
  }

  // CSGGXDiffusePhaseFunction

  CSGGXDiffusePhaseFunction::CSGGXDiffusePhaseFunction(const SSGGXDistributionParameters& distributionParameters) :
    m_distribution(distributionParameters) {

  }

  float CSGGXDiffusePhaseFunction::p(const glm::vec3& w_o, const glm::vec3& w_i) const {
    glm::vec3 w_m = glm::normalize(w_i + w_o);
    return glm::max(glm::dot(w_o, w_m), 0.f) / M_PI;
  }

  float CSGGXDiffusePhaseFunction::sampleP(const glm::vec3& wo, glm::vec3* wi, CSampler& sampler) const {
    glm::vec3 w_n = m_distribution.sampleVNDF(wo, glm::vec2(sampler.uniformSample01(), sampler.uniformSample01()));
    CCoordinateFrame frame = CCoordinateFrame::fromNormal(w_n);
    glm::vec3 w_m = sampler.uniformSampleHemisphere();
    glm::vec3 w_m_world = frame.tangentToWorld() * glm::vec4(w_m, 0.f);
    *wi = glm::normalize(w_m_world); // TODO: Check coordinate space
    return p(wo, *wi);
  }

  // CSGGXPhaseFunction

  CSGGXPhaseFunction::CSGGXPhaseFunction(const SSGGXDistributionParameters& distributionParameters) :
    CPhaseFunction(EPhaseFunction::SGGX),
    m_specular(distributionParameters),
    m_diffuse(distributionParameters) {

  }

  float CSGGXPhaseFunction::p(const glm::vec3& w_o, const glm::vec3& w_i) const {
    return m_specular.p(w_o, w_i) + m_diffuse.p(w_o, w_i);
  }

  float CSGGXPhaseFunction::sampleP(const glm::vec3& wo, glm::vec3* wi, CSampler& sampler) const {
    m_specular.sampleP(wo, wi, sampler);
    return p(wo, *wi);
  }
}