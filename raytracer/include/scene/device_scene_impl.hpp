#ifndef DEVICE_SCENE_IMPL_HPP
#define DEVICE_SCENE_IMPL_HPP
#include "device_scene.hpp"
#include "medium/medium_instance_impl.hpp"
#include "device_sceneobject.hpp"
#include "sampling/distribution_1d.hpp"
#include "scene/environmentmap.hpp"
#include "intersect/ray.hpp"

namespace rt {

  inline void CDeviceScene::intersect(const CRay& ray, SInteraction* closestInteraction, OptixVisibilityMask visibilityMask) const {
    unsigned int siAdress[2];
    memcpy(siAdress, &closestInteraction, sizeof(SInteraction*));
    optixTrace(m_traversableHandle,
      float3{ ray.m_origin.x, ray.m_origin.y, ray.m_origin.z },
      float3{ ray.m_direction.x, ray.m_direction.y, ray.m_direction.z },
      0.f,
      ray.m_t_max,
      0.f,
      visibilityMask,
      0,
      0,
      1,
      0,
      siAdress[0],
      siAdress[1]);
    if (closestInteraction->hitInformation.hit) {
      ray.m_t_max = closestInteraction->hitInformation.t;
    }
  }

  inline glm::vec3 CDeviceScene::sampleLightSources(CSampler& sampler, glm::vec3* direction, float* pdf) const {
    if (m_envMap) {
      return m_envMap->sample(sampler, direction, pdf);
    }
    return glm::vec3(0.f);
  }

  inline glm::vec3 CDeviceScene::le(const glm::vec3& direction, float* pdf) const {
    if (m_envMap) {
      return m_envMap->le(direction, pdf);
    }
    return glm::vec3(0.f);
  }

  inline float CDeviceScene::lightSourcesPdf(const SInteraction& lightHit) const {
    if (lightHit.object) {
      float power = lightHit.object->power();
      float totalPower = m_lightDist->integral();
      return power / totalPower;
    }
    return 0.0f;
  }

  inline float CDeviceScene::lightSourcePdf(const SInteraction& lightHit, const CRay& shadowRay) const {
    if (lightHit.object) {
      const CShape* lightGeometry = lightHit.object->shape();
      switch (lightGeometry->shape()) {
      case EShape::CIRCLE: {

        return ((const CCircle*)lightGeometry)->pdf(lightHit, shadowRay);
      }
      }
    }
    return 0.0f;
  }


  inline glm::vec3 CDeviceScene::tr(const CRay& ray, CSampler& sampler, bool useBrickGrid) const {
    glm::vec3 p0 = ray.m_origin;
    const glm::vec3 p1 = p0 + ray.m_t_max * ray.m_direction;
    const CMediumInstance* currentMedium = ray.m_medium;
    glm::vec3 Tr(1.f);
    while (true) {
      CRay r = CRay::spawnRay(p0, p1, currentMedium);
      SInteraction interaction;
      intersect(r, &interaction);
      if (interaction.hitInformation.hit && r.m_medium) {
        Tr *= r.m_medium->tr(r, sampler, useBrickGrid);
      }

      if (!interaction.hitInformation.hit) {
        break;
      }

      p0 = interaction.hitInformation.pos;
      currentMedium = !r.m_medium ? interaction.medium : nullptr;
    }
    return Tr;
  }

  inline SInteraction CDeviceScene::intersectTr(const CRay& ray, CSampler& sampler, glm::vec3* Tr, bool useBrickGrid) const {
    *Tr = glm::vec3(1.f);
    glm::vec3 p0 = ray.m_origin;
    const glm::vec3 p1 = p0 + ray.m_t_max * ray.m_direction;
    const CMediumInstance* currentMedium = ray.m_medium;
    while (true) {
      CRay r = CRay::spawnRay(p0, p1, currentMedium);

      SInteraction interaction;
      intersect(r, &interaction);
      if (interaction.hitInformation.hit && r.m_medium) {
        *Tr *= r.m_medium->tr(r, sampler, useBrickGrid);
      }

      if (!interaction.hitInformation.hit || interaction.material) {
        return interaction;
      }

      p0 = interaction.hitInformation.pos;
      currentMedium = !r.m_medium ? interaction.medium : nullptr;
    }
  }

  inline void CDeviceScene::intersectTr(const CRay& ray, CSampler& sampler, glm::vec3* Tr, SInteraction* interaction, bool useBrickGrid) const {
    *Tr = glm::vec3(1.f);
    glm::vec3 p0 = ray.m_origin;
    const glm::vec3 p1 = p0 + ray.m_t_max * ray.m_direction;
    const CMediumInstance* currentMedium = ray.m_medium;
    while (true) {
      CRay r = CRay::spawnRay(p0, p1, currentMedium);

      intersect(r, interaction);
      if (interaction->hitInformation.hit && r.m_medium) {
        *Tr *= r.m_medium->tr(r, sampler, useBrickGrid);
      }

      if (!interaction->hitInformation.hit || interaction->material) {
        return;
      }

      p0 = interaction->hitInformation.pos;
      currentMedium = !r.m_medium ? interaction->medium : nullptr;
    }
  }
}
#endif