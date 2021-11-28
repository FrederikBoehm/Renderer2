#include "medium/medium.hpp"

#include "medium/homogeneous_medium.hpp"
#include <stdio.h>
#include "intersect/ray.hpp"
#include "sampling/sampler.hpp"
#include "scene/interaction.hpp"
#include "medium/heterogenous_medium.hpp"
#include "medium/nvdb_medium.hpp"
#include "medium/phase_function.hpp"

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
    case EMediumType::HETEROGENOUS_MEDIUM:
      return ((CHeterogenousMedium*)this)->tr(ray, sampler);
    case EMediumType::NVDB_MEDIUM:
      return ((CNVDBMedium*)this)->tr(ray, sampler);
    }
    printf("No matching medium\n");
    return glm::vec3(0.f);
  }

  glm::vec3 CMedium::sample(const CRay& ray, CSampler& sampler, SInteraction* mi) const {
    switch (m_type) {
    case EMediumType::HOMOGENEOUS_MEDIUM:
      return ((CHomogeneousMedium*)this)->sample(ray, sampler, mi);
    case EMediumType::HETEROGENOUS_MEDIUM:
      return ((CHeterogenousMedium*)this)->sample(ray, sampler, mi);
    case EMediumType::NVDB_MEDIUM:
      return ((CNVDBMedium*)this)->sample(ray, sampler, mi);
    }
    printf("No matching medium\n");
    return glm::vec3(0.f);
  }

  const CPhaseFunction& CMedium::phase() const {
    switch (m_type) {
    case EMediumType::HOMOGENEOUS_MEDIUM:
      return ((CHomogeneousMedium*)this)->phase();
    case EMediumType::HETEROGENOUS_MEDIUM:
      return ((CHeterogenousMedium*)this)->phase();
    case EMediumType::NVDB_MEDIUM:
      return ((CNVDBMedium*)this)->phase();
    }
    printf("No matching medium\n");
    return CHenyeyGreensteinPhaseFunction(0.f);
  }

  glm::vec3 CMedium::normal(const glm::vec3& p, CSampler& sampler) const {
    switch (m_type) {
    case EMediumType::NVDB_MEDIUM:
      return ((CNVDBMedium*)this)->normal(p, sampler);
    }
    return glm::vec3(1.f, 0.f, 0.f);
  }
}