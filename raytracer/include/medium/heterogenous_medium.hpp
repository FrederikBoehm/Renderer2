#ifndef HETEROGENOUS_MEDIUM_HPP
#define HETEROGENOUS_MEDIUM_HPP

#include "medium.hpp"
namespace rt {
  struct SHeterogenousMedium_DeviceResource {
    float* d_density;
  };

  class CHeterogenousMedium : public CMedium {
  public:
    H_CALLABLE CHeterogenousMedium(const glm::vec3& sigma_a, const glm::vec3& sigma_s, float g, uint32_t nx, uint32_t ny, uint32_t nz, const glm::vec3& pos, const glm::vec3& size, const float* d);

    DH_CALLABLE float density(const glm::vec3& p) const;
    DH_CALLABLE float D(const glm::ivec3& p) const;
    DH_CALLABLE glm::vec3 sample(const CRay& rayWorld, CSampler& sampler, SInteraction* mi) const;
    DH_CALLABLE glm::vec3 tr(const CRay& ray, CSampler& sampler) const;

    DH_CALLABLE const CHenyeyGreensteinPhaseFunction& phase() const;

    H_CALLABLE void allocateDeviceMemory();
    H_CALLABLE CHeterogenousMedium copyToDevice() const;
    H_CALLABLE void freeDeviceMemory() const;

  private:
    const glm::vec3 m_sigma_a;
    const glm::vec3 m_sigma_s;
    //const float m_g;
    const CHenyeyGreensteinPhaseFunction m_phase;
    const uint32_t m_nx;
    const uint32_t m_ny;
    const uint32_t m_nz;
    const glm::mat4 m_mediumToWorld;
    const glm::mat4 m_worldToMedium;
    float* m_density;

    float m_sigma_t;
    float m_invMaxDensity;

    SHeterogenousMedium_DeviceResource* m_deviceResource;

    DH_CALLABLE static glm::mat4 getMediumToWorldTransformation(const glm::vec3& pos, const glm::vec3& size);

    H_CALLABLE CHeterogenousMedium(const glm::vec3& sigma_a,
                                   const glm::vec3& sigma_s,
                                   const CHenyeyGreensteinPhaseFunction& phase, 
                                   const uint32_t nx, 
                                   const uint32_t ny,
                                   const uint32_t nz,
                                   const glm::mat4& mediumToWorld,
                                   const glm::mat4& worldToMedium,
                                   float* density,
                                   float sigma_t,
                                   float invMaxDensity,
                                   SHeterogenousMedium_DeviceResource* deviceResource);
  };

  inline const CHenyeyGreensteinPhaseFunction& CHeterogenousMedium::phase() const {
    return m_phase;
  }
}
#endif // !HETEROGENOUS_MEDIUM_HPP
