#include "medium/medium.hpp"

#include "medium/homogeneous_medium.hpp"
#include <stdio.h>
#include "intersect/ray.hpp"
#include "sampling/sampler.hpp"
#include "scene/interaction.hpp"
#include "medium/heterogenous_medium.hpp"

namespace rt {
  CMedium::CMedium(const EMediumType type) :
    m_type(type) {

  }

  CMedium::~CMedium() {

  }

  glm::vec3 CMedium::tr(const CRay& ray, CSampler& sampler) const {
    switch (m_type) {
    case EMediumType::HOMOGENEOUS_MEDIUM:
      return ((CHomogeneousMedium*)this)->tr(ray, sampler);
      break;
    case EMediumType::HETEROGENOUS_MEDIUM:
      return ((CHeterogenousMedium*)this)->tr(ray, sampler);
    }
    printf("No matching medium\n");
    return glm::vec3(0.f);
  }

  glm::vec3 CMedium::sample(const CRay& ray, CSampler& sampler, SInteraction* mi) const {
    switch (m_type) {
    case EMediumType::HOMOGENEOUS_MEDIUM:
      return ((CHomogeneousMedium*)this)->sample(ray, sampler, mi);
      break;
    case EMediumType::HETEROGENOUS_MEDIUM:
      return ((CHeterogenousMedium*)this)->sample(ray, sampler, mi);
      break;
    }
    printf("No matching medium\n");
    return glm::vec3(0.f);
  }

  const CHenyeyGreensteinPhaseFunction& CMedium::phase() const {
    switch (m_type) {
    case EMediumType::HOMOGENEOUS_MEDIUM:
      return ((CHomogeneousMedium*)this)->phase();
      break;
    case EMediumType::HETEROGENOUS_MEDIUM:
      return ((CHeterogenousMedium*)this)->phase();
    }
    printf("No matching medium\n");
    return CHenyeyGreensteinPhaseFunction(0.f);
  }
}