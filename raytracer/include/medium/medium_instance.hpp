#ifndef MEDIUM_INSTANCE_HPP
#define MEDIUM_INSTANCE_HPP
#include "utility/qualifiers.hpp"
#include <glm/glm.hpp>
#include "intersect/aabb.hpp"
#include <string>
#include <optix/optix_types.h>
namespace rt {
  class CNVDBMedium;
  class CPhaseFunction;
  class CRay;
  class SInteraction;
  class CSampler;

  // Adaptor for NanoVDB Media: Transforms between world and model space
  class CMediumInstance {
  public:
    H_CALLABLE CMediumInstance(CNVDBMedium* medium, const glm::mat4x3* modelToWorld, const glm::mat4x3* worldToModel);

    D_CALLABLE glm::vec3 sample(const CRay& rayWorld, CSampler& sampler, SInteraction* mi, bool useBrickGrid, size_t* numLookups) const;
    D_CALLABLE glm::vec3 tr(const CRay& ray, CSampler& sampler, bool useBrickGrid, size_t* numLookups) const;
    DH_CALLABLE const CPhaseFunction& phase() const;
    DH_CALLABLE SAABB worldBB() const;
    DH_CALLABLE SAABB modelBB() const;
    D_CALLABLE CRay moveToVoxelBorder(const CRay& ray) const;
    
    H_CALLABLE std::string path() const;
    H_CALLABLE OptixTraversableHandle getOptixHandle() const;
    H_CALLABLE OptixProgramGroup getOptixProgramGroup() const;
    H_CALLABLE void setFilterRenderRatio(float ratio);
    H_CALLABLE float filterRenderRatio() const;
    DH_CALLABLE float voxelSizeFiltering() const;
  private:
    CNVDBMedium* m_medium;
    const glm::mat4x3* m_modelToWorld; // Pointer to sceneobject's modelToWorld transform
    const glm::mat4x3* m_worldToModel; // Pointer to sceneobject's worldToModel transform
    float m_filterRenderRatio;
  };

  
}
#endif // !MEDIUM_INSTANCE_HPP
