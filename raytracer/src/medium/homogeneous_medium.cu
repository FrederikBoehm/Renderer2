#include "medium/homogeneous_medium.hpp"
#include "intersect/ray.hpp"
#include <algorithm>
#include "scene/interaction.hpp"
#include "sampling/sampler.hpp"
#include "scene/interaction.hpp"

namespace rt {
  CHomogeneousMedium::CHomogeneousMedium(const glm::vec3& sigma_a, const glm::vec3& sigma_s, float g) :
    CMedium(EMediumType::HOMOGENEOUS_MEDIUM),
    m_sigma_a(sigma_a),
    m_sigma_s(sigma_s),
    m_sigma_t(sigma_s + sigma_a),
    m_phase(g) {

  }

  glm::vec3 CHomogeneousMedium::tr(const CRay& ray, const CSampler& sampler) const {
    return glm::exp(-m_sigma_t * glm::min(ray.m_t_max * glm::length(ray.m_direction), FLT_MAX));
  }

  glm::vec3 CHomogeneousMedium::sample(const CRay& ray, CSampler& sampler, SInteraction* mi) const {
    uint16_t channel = sampler.uniformSample01() * 3;
    float dist = -glm::log(1 - sampler.uniformSample01()) / m_sigma_t[channel];
    float t = glm::min(dist * glm::length(ray.m_direction), ray.m_t_max);
    bool sampledMedium = t < ray.m_t_max;
    if (sampledMedium) {
      SHitInformation hit = { true, ray.m_origin + t * ray.m_direction , glm::vec3(0.0f), t };
      //*mi = {};
      *mi = { hit, nullptr, nullptr, this };

    }
    glm::vec3 Tr = glm::exp(-m_sigma_t * glm::min(t, FLT_MAX) * glm::length(ray.m_direction));

    glm::vec3 density = sampledMedium ? (m_sigma_t * Tr) : Tr;
    float pdf = 0;
    for (size_t i = 0; i < 3; ++i) {
      pdf += density[i];
    }
    pdf /= 3;
    return sampledMedium ? (Tr * m_sigma_s / pdf) : (Tr / pdf);
  }
}