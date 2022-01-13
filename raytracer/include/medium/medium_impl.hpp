#ifndef MEDIUM_IMPL_HPP
#define MEDIUM_IMPL_HPP
#include "medium.hpp"
#include "nvdb_medium_impl.hpp"
namespace rt {
  

  

  inline glm::vec3 CMedium::tr(const CRay& ray, CSampler& sampler) const {
    switch (m_type) {
    //case EMediumType::HOMOGENEOUS_MEDIUM:
    //  return ((CHomogeneousMedium*)this)->tr(ray, sampler);
    //case EMediumType::HETEROGENOUS_MEDIUM:
    //  return ((CHeterogenousMedium*)this)->tr(ray, sampler);
    case EMediumType::NVDB_MEDIUM:
      return ((CNVDBMedium*)this)->tr(ray, sampler);
    }
    printf("[ERROR]: No matching medium\n");
    return glm::vec3(0.f);
  }

  inline glm::vec3 CMedium::sample(const CRay& ray, CSampler& sampler, SInteraction* mi) const {
    switch (m_type) {
    //case EMediumType::HOMOGENEOUS_MEDIUM:
    //  return ((CHomogeneousMedium*)this)->sample(ray, sampler, mi);
    //case EMediumType::HETEROGENOUS_MEDIUM:
    //  return ((CHeterogenousMedium*)this)->sample(ray, sampler, mi);
    case EMediumType::NVDB_MEDIUM:
      //return ((CNVDBMedium*)this)->sample(ray, sampler, mi);
      return static_cast<const CNVDBMedium*>(this)->sample(ray, sampler, mi);
    }
    printf("[ERROR]: No matching medium\n");
    return glm::vec3(0.f);
  }

  inline const CPhaseFunction& CMedium::phase() const {
    switch (m_type) {
    //case EMediumType::HOMOGENEOUS_MEDIUM:
    //  return ((CHomogeneousMedium*)this)->phase();
    //case EMediumType::HETEROGENOUS_MEDIUM:
    //  return ((CHeterogenousMedium*)this)->phase();
    case EMediumType::NVDB_MEDIUM:
      return ((CNVDBMedium*)this)->phase();
    }
    printf("[ERROR]: No matching medium\n");
    return CHenyeyGreensteinPhaseFunction(0.f);
  }

  inline glm::vec3 CMedium::normal(const glm::vec3& p, CSampler& sampler) const {
    switch (m_type) {
    case EMediumType::NVDB_MEDIUM:
      return ((CNVDBMedium*)this)->normal(p, sampler);
    }
    return glm::vec3(1.f, 0.f, 0.f);
  }

  DH_CALLABLE inline EMediumType CMedium::type() const {
    return m_type;
  }

}
#endif