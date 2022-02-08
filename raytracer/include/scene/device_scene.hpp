#ifndef DEVICE_SCENE_HPP
#define DEVICE_SCENE_HPP
#include "utility/qualifiers.hpp"
#include <optix/optix_device.h>
#include "interaction.hpp"
namespace rt {
  class CHostScene;
  class CDistribution1D;
  class CEnvironmentMap;
  class CDeviceSceneobject;
  class CRay;

  class CDeviceScene {
    friend class CSceneConnection;
    friend struct SSharedMemoryInitializer;
  public:
    D_CALLABLE void intersect(const CRay& ray, SInteraction* closestInteraction, OptixVisibilityMask visibilityMask = 255) const;
    D_CALLABLE glm::vec3 sampleLightSources(CSampler& sampler, glm::vec3* direction, float* pdf) const;
    D_CALLABLE glm::vec3 le(const glm::vec3& direction, float* pdf) const;
    D_CALLABLE float lightSourcePdf(const SInteraction& lightHit, const CRay& shadowRay) const;
    D_CALLABLE float lightSourcesPdf(const SInteraction& lightHit) const;

    D_CALLABLE glm::vec3 tr(const CRay& ray, CSampler& sampler) const;
    D_CALLABLE SInteraction intersectTr(const CRay& ray, CSampler& sampler, glm::vec3* Tr) const;
    D_CALLABLE void intersectTr(const CRay& ray, CSampler& sampler, glm::vec3* Tr, SInteraction* interaction) const;

  private:
    uint16_t m_numSceneobjects;
    CDeviceSceneobject* m_sceneobjects;
    uint16_t m_numLights;
    CDeviceSceneobject* m_lights; // For now lights are also sceneobjects
    CDistribution1D* m_lightDist;
    CEnvironmentMap* m_envMap;
    OptixTraversableHandle m_traversableHandle;
  };

  
}
#endif // !DEVICE_SCENE_HPP
