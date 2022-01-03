#ifndef OREN_NAYAR_BRDF_HPP
#define OREN_NAYAR_BRDF_HPP
#define _USE_MATH_DEFINES
#include <math.h>
#include <glm/glm.hpp>

#include "utility/qualifiers.hpp"
#include "intersect/hit_information.hpp"
#include "brdf_functions.hpp"
#include "sampling/sampler.hpp"

namespace rt {
  // Better approximation of rough surfaces than lambertian reflection
  class COrenNayarBRDF {
  public:
    DH_CALLABLE COrenNayarBRDF();
    DH_CALLABLE COrenNayarBRDF(float roughness);

    // Expects incident and outgoing directions in tangent space
    DH_CALLABLE glm::vec3 f(const glm::vec3& wo, const glm::vec3& wi) const;
    D_CALLABLE glm::vec3 sampleF(const glm::vec3& wo, glm::vec3* wi, CSampler& sampler, float* pdf) const;
    D_CALLABLE float pdf(const glm::vec3& wo, const glm::vec3& wi) const;

  private:
    float m_A;
    float m_B;
  };

  inline COrenNayarBRDF::COrenNayarBRDF() :
    m_A(0.0f),
    m_B(0.0f) {

  }

  inline COrenNayarBRDF::COrenNayarBRDF(float roughness) {
    float sigma2 = roughness * roughness;
    m_A = 1.0f - (sigma2 / (2.0f * (sigma2 + 0.33f)));
    m_B = 0.45f * sigma2 / (sigma2 + 0.09f);
  }

  inline glm::vec3 COrenNayarBRDF::f(const glm::vec3& wo, const glm::vec3& wi) const {
    float sinThetaI = sinTheta(wi);
    float sinThetaO = sinTheta(wo);

    float maxCos = 0.0f;
    if (sinThetaI > 1.0e-4 && sinThetaO > 1.0e-4) {
      float sinPhiI = sinPhi(wi);
      float cosPhiI = cosPhi(wi);
      float sinPhiO = sinPhi(wo);
      float cosPhiO = cosPhi(wo);
      float dCos = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
      maxCos = glm::max(0.0f, dCos);
    }

    float sinAlpha = 0.0f;
    float tanBeta = 0.0f;
    if (absCosTheta(wi) > absCosTheta(wo)) {
      sinAlpha = sinThetaO;
      tanBeta = sinThetaI / absCosTheta(wi);
    }
    else {
      sinAlpha = sinThetaI;
      tanBeta = sinThetaO / absCosTheta(wo);
    }

    return glm::vec3(M_1_PI) * (m_A + m_B * maxCos * sinAlpha * tanBeta);
  }

  inline glm::vec3 COrenNayarBRDF::sampleF(const glm::vec3& wo, glm::vec3* wi, CSampler& sampler, float* pdf) const {
    *wi = sampler.cosineSampleHemisphere();
    *pdf = CSampler::cosineHemispherePdf(cosTheta(*wi));
    return f(wo, *wi);
  }

  inline float COrenNayarBRDF::pdf(const glm::vec3& wo, const glm::vec3& wi) const {
    return CSampler::cosineHemispherePdf(cosTheta(wi));
  }
}
#endif

