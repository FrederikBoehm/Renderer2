#ifndef MEDIUM_INSTANCE_HPP
#define MEDIUM_INSTANCE_HPP
#include "utility/qualifiers.hpp"
#include <glm/glm.hpp>
#include "intersect/aabb.hpp"
#include <string>
namespace rt {
  class CNVDBMedium;
  class CPhaseFunction;
  class CRay;
  class SInteraction;

  // Adaptor for NanoVDB Media: Transforms between world and model space
  class CMediumInstance {
  public:
    H_CALLABLE CMediumInstance(CNVDBMedium* medium, const glm::mat4x3* modelToWorld, const glm::mat4x3* worldToModel);

    D_CALLABLE glm::vec3 sample(const CRay& rayWorld, CSampler& sampler, SInteraction* mi) const;
    D_CALLABLE glm::vec3 tr(const CRay& ray, CSampler& sampler) const;
    D_CALLABLE glm::vec3 normal(const glm::vec3& p, CSampler& sampler) const;
    DH_CALLABLE const CPhaseFunction& phase() const;
    DH_CALLABLE SAABB worldBB() const;
    
    H_CALLABLE std::string path() const;
    H_CALLABLE OptixTraversableHandle getOptixHandle() const;
    H_CALLABLE OptixProgramGroup getOptixProgramGroup() const;
  private:
    CNVDBMedium* m_medium;
    const glm::mat4x3* m_modelToWorld; // Pointer to sceneobject's modelToWorld transform
    const glm::mat4x3* m_worldToModel; // Pointer to sceneobject's worldToModel transform
  };

  
}
#endif // !MEDIUM_INSTANCE_HPP
