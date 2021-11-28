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
    m_invS(glm::inverse(m_S)),
    m_sigma(0.f) {

  }

  CSGGXMicroflakeDistribution::CSGGXMicroflakeDistribution(float sigma) :
    m_S(1.f),
    m_invS(1.f),
    m_sigma(sigma) {

  }

  glm::mat3 CSGGXMicroflakeDistribution::getS(const SSGGXDistributionParameters& parameters) {
    glm::mat3 S(parameters.m_Sxx, parameters.m_Sxy, parameters.m_Sxz,
                parameters.m_Sxy, parameters.m_Syy, parameters.m_Syz,
                parameters.m_Sxz, parameters.m_Syz, parameters.m_Szz);
    return S;
  }

  glm::mat3 CSGGXMicroflakeDistribution::buildS(const glm::vec3& n) const {
    float xx = n.x * n.x;
    float xy = n.x * n.y;
    float xz = n.x * n.z;
    float yy = n.y * n.y;
    float yz = n.y * n.z;
    float zz = n.z * n.z;
    glm::mat3 m1(xx, xy, xz, xy, yy, yz, xz, yz, zz);
    glm::mat3 m2(yy + zz, -xy, -xz, -xy, xx + zz, -yz, -xz, -yz, xx + yy);
    return m1 + m_sigma * m2;
  }

  float CSGGXMicroflakeDistribution::D(const glm::vec3& w) const {
    float v = glm::dot(w, m_invS * w);
    return 1.f / (M_PI * glm::sqrt(glm::abs(glm::determinant(m_S))) * v * v);
  }

  float CSGGXMicroflakeDistribution::D(const glm::vec3& w, const glm::mat3& S) const {
    glm::mat3 invS = glm::inverse(S);
    float v = glm::dot(w, invS * w);
    return 1.f / (M_PI * glm::sqrt(glm::abs(glm::determinant(S))) * v * v);
  }

  float CSGGXMicroflakeDistribution::sigma(const glm::vec3& w) const {
    return glm::sqrt(glm::max(glm::dot(w, m_S * w), 0.f));
  }

  float CSGGXMicroflakeDistribution::sigma(const glm::vec3& w, const glm::mat3& S) const {
    return glm::sqrt(glm::max(glm::dot(w, S * w), 0.f));
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

  glm::vec3 CSGGXMicroflakeDistribution::sampleVNDF(const glm::vec3& w_i, const glm::vec2& U, const glm::mat3& S) const {
    const CCoordinateFrame frame = CCoordinateFrame::fromNormal(w_i);
    const glm::vec3& w_k = frame.B();
    const glm::vec3& w_j = frame.T();

    const glm::mat3 S_kji = projectS(S, w_k, w_j, w_i);

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

  CSGGXSpecularPhaseFunction::CSGGXSpecularPhaseFunction(float sigma) :
    m_distribution(sigma) {

  }

  float CSGGXSpecularPhaseFunction::p(const glm::vec3& w_o, const glm::vec3& w_i, const glm::vec3& n) const {
    const glm::vec3 w_h = glm::normalize(w_i + w_o);
    const glm::mat3 S = m_distribution.buildS(n);
    return m_distribution.D(w_h, S) / (4.f * m_distribution.sigma(w_o, S)); // w_o because of different convention for incoming and outgoing directions
  }

  float CSGGXSpecularPhaseFunction::sampleP(const glm::vec3& wo, glm::vec3* wi, const glm::vec3& n, CSampler& sampler) const {
    glm::vec3 w_m = m_distribution.sampleVNDF(wo, glm::vec2(sampler.uniformSample01(), sampler.uniformSample01()), m_distribution.buildS(n));
    *wi = glm::reflect(-wo, w_m);
    return p(wo, *wi, n);
  }

  // CSGGXDiffusePhaseFunction

  CSGGXDiffusePhaseFunction::CSGGXDiffusePhaseFunction(const SSGGXDistributionParameters& distributionParameters) :
    m_distribution(distributionParameters) {

  }

  CSGGXDiffusePhaseFunction::CSGGXDiffusePhaseFunction(float sigma) :
    m_distribution(sigma) {

  }

  float CSGGXDiffusePhaseFunction::p(const glm::vec3& w_o, const glm::vec3& w_i, const glm::vec3& normal, CSampler& sampler) const {
    glm::vec3 w_m = m_distribution.sampleVNDF(w_o, glm::vec2(sampler.uniformSample01(), sampler.uniformSample01()), m_distribution.buildS(normal));
    return glm::max(glm::dot(w_i, w_m), 0.f) / M_PI;
  }

  float CSGGXDiffusePhaseFunction::p(const glm::vec3& w_o, const glm::vec3& w_i, const glm::vec3& normal) const {
    return glm::max(glm::dot(w_i, normal), 0.f) / M_PI;
  }

  float CSGGXDiffusePhaseFunction::sampleP(const glm::vec3& wo, glm::vec3* wi, const glm::vec3& n, CSampler& sampler) const {
    glm::vec3 w_n = m_distribution.sampleVNDF(wo, glm::vec2(sampler.uniformSample01(), sampler.uniformSample01()), m_distribution.buildS(n));
    CCoordinateFrame frame = CCoordinateFrame::fromNormal(w_n);
    glm::vec3 w_m = sampler.uniformSampleHemisphere();
    glm::vec3 w_m_world = frame.tangentToWorld() * glm::vec4(w_m, 0.f);
    *wi = glm::normalize(w_m_world); // TODO: Check coordinate space
    return p(wo, *wi, w_n);
  }

  // CSGGXPhaseFunction

  CSGGXPhaseFunction::CSGGXPhaseFunction(const SSGGXDistributionParameters& diffuseParameters, const SSGGXDistributionParameters& specularParameters) :
    CPhaseFunction(EPhaseFunction::SGGX),
    m_diffuse(diffuseParameters),
    m_specular(specularParameters) {

  }

  CSGGXPhaseFunction::CSGGXPhaseFunction(float diffuseSigma, float specularSigma) :
    CPhaseFunction(EPhaseFunction::SGGX),
    m_diffuse(diffuseSigma),
    m_specular(specularSigma) {

  }

  float CSGGXPhaseFunction::p(const glm::vec3& w_o, const glm::vec3& w_i, const glm::vec3& n, CSampler& sampler) const {
    if (sampler.uniformSample01() < 0.5f) {
      return 2.f * m_diffuse.p(w_o, w_i, n, sampler);
    }
    else {
      return 2.f * m_specular.p(w_o, w_i, n);
    }
  }

  float CSGGXPhaseFunction::sampleP(const glm::vec3& wo, glm::vec3* wi, const glm::vec3& n, CSampler& sampler) const {
    if (sampler.uniformSample01() < 0.5f) {
      return 2.f * m_diffuse.sampleP(wo, wi, n, sampler);
    }
    else {
      return 2.f * m_specular.sampleP(wo, wi, n, sampler);
    }
  }
}