#define _USE_MATH_DEFINES
#include <math.h>
#include "material/oren_nayar_brdf.hpp"
#include "material/brdf_functions.hpp"
#include <iostream>

namespace rt {
  COrenNayarBRDF::COrenNayarBRDF():
    m_albedo(0.0f),
    m_A(0.0f),
    m_B(0.0f) {

  }

  COrenNayarBRDF::COrenNayarBRDF(const glm::vec3& albedo, float roughness):
    m_albedo(albedo){
    float sigma2 = roughness * roughness;
    m_A = 1.0f - (sigma2 / (2.0f * (sigma2 + 0.33f)));
    m_B = 0.45f * sigma2 / (sigma2 + 0.09f);
  }

  glm::vec3 COrenNayarBRDF::f(const SHitInformation& hitInformation, const glm::vec3& wo, const glm::vec3& wi) const {
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

    //printf("Albedo: r: %f, g: %f, b: %f, m_A: %f, m_B: %f, maxCos: %f, sinAlpha: %f, tanBeta: %f\n", m_albedo.r, m_albedo.g, m_albedo.b, m_A, m_B, maxCos, sinAlpha, tanBeta);
    return m_albedo * glm::vec3(M_1_PI) * (m_A + m_B * maxCos * sinAlpha * tanBeta);
  }
}