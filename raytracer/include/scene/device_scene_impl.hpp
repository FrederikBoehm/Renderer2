#ifndef DEVICE_SCENE_IMPL_HPP
#define DEVICE_SCENE_IMPL_HPP
#include "device_scene.hpp"
#include "medium/medium_impl.hpp"
#include "device_sceneobject.hpp"
#include "sampling/distribution_1d.hpp"
#include "scene/environmentmap.hpp"

namespace rt {
  //inline SInteraction CDeviceScene::intersect(const CRay& ray) const {
  //  SInteraction closestInteraction;
  //  closestInteraction.hitInformation.hit = false;
  //  closestInteraction.hitInformation.t = CRay::DEFAULT_TMAX;
  //  closestInteraction.object = nullptr;
  //  closestInteraction.material = nullptr;
  //  closestInteraction.medium = nullptr;
  //  SInteraction* siPtr = &closestInteraction;
  //  unsigned int siAdress[2];
  //  memcpy(siAdress, &siPtr, sizeof(SInteraction*));
  //  optixTrace(m_traversableHandle,
  //    float3{ ray.m_origin.x, ray.m_origin.y, ray.m_origin.z },
  //    float3{ ray.m_direction.x, ray.m_direction.y, ray.m_direction.z },
  //    0.f,
  //    ray.m_t_max,
  //    0.f,
  //    255,
  //    0,
  //    0,
  //    1,
  //    0,
  //    siAdress[0],
  //    siAdress[1]);
  //  if (closestInteraction.hitInformation.hit) {
  //    ray.m_t_max = closestInteraction.hitInformation.t;
  //  }
  //  return closestInteraction;
  //}

  inline void CDeviceScene::intersect(const CRay& ray, SInteraction* closestInteraction) const {
    unsigned int siAdress[2];
    memcpy(siAdress, &closestInteraction, sizeof(SInteraction*));
    if (glm::any(glm::isinf(ray.m_origin)) || glm::any(glm::isnan(ray.m_origin)) ||
      glm::any(glm::isinf(ray.m_direction)) || glm::any(glm::isnan(ray.m_direction)) ||
      glm::isinf(ray.m_t_max) || glm::isnan(ray.m_t_max)) {
      uint3 launchIdx = optixGetLaunchIndex();
      glm::vec3 pos = ray.m_origin;
      glm::vec3 n = ray.m_direction;
      printf("origin: (%f, %f, &f), direction (%f, %f, %f), t %f at (%i, %i)\n", pos.x, pos.y, pos.z, n.x, n.y, n.z, ray.m_t_max, (int)launchIdx.x, (int)launchIdx.y);
    }
    optixTrace(m_traversableHandle,
      float3{ ray.m_origin.x, ray.m_origin.y, ray.m_origin.z },
      float3{ ray.m_direction.x, ray.m_direction.y, ray.m_direction.z },
      0.f,
      ray.m_t_max,
      0.f,
      255,
      0,
      0,
      1,
      0,
      siAdress[0],
      siAdress[1]);
    if (glm::any(glm::isinf(closestInteraction->hitInformation.pos)) || glm::any(glm::isnan(closestInteraction->hitInformation.pos)) ||
      glm::any(glm::isinf(closestInteraction->hitInformation.normal)) || glm::any(glm::isnan(closestInteraction->hitInformation.normal)) ||
      glm::isinf(closestInteraction->hitInformation.t) || glm::isnan(closestInteraction->hitInformation.t)) {
      uint3 launchIdx = optixGetLaunchIndex();
      glm::vec3 pos = closestInteraction->hitInformation.pos;
      glm::vec3 n = closestInteraction->hitInformation.normal;
      printf("origin: (%f, %f, &f), normal (%f, %f, %f) at (%i, %i)\n", pos.x, pos.y, pos.z, n.x, n.y, n.z, closestInteraction->hitInformation.t, (int)launchIdx.x, (int)launchIdx.y);
    }
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

  //inline bool CDeviceScene::occluded(const CRay& ray) const {
  //  return intersect(ray).hitInformation.hit;
  //}

  inline glm::vec3 CDeviceScene::tr(const CRay& ray, CSampler& sampler) const {
    glm::vec3 p0 = ray.m_origin;
    const glm::vec3 p1 = p0 + ray.m_t_max * ray.m_direction;
    const CMedium* currentMedium = ray.m_medium;
    glm::vec3 Tr(1.f);
    while (true) {
      CRay r = CRay::spawnRay(p0, p1, currentMedium);
      SInteraction interaction;
      intersect(r, &interaction);
      //SInteraction interaction = intersect(r);
      if (interaction.hitInformation.hit && r.m_medium) {
        Tr *= r.m_medium->tr(r, sampler);
      }

      if (!interaction.hitInformation.hit) {
        break;
      }

      p0 = interaction.hitInformation.pos;
      currentMedium = !r.m_medium ? interaction.medium : nullptr;
    }
    return Tr;
  }

  inline SInteraction CDeviceScene::intersectTr(const CRay& ray, CSampler& sampler, glm::vec3* Tr) const {
    *Tr = glm::vec3(1.f);
    glm::vec3 p0 = ray.m_origin;
    const glm::vec3 p1 = p0 + ray.m_t_max * ray.m_direction;
    const CMedium* currentMedium = ray.m_medium;
    while (true) {
      CRay r = CRay::spawnRay(p0, p1, currentMedium);

      SInteraction interaction;
      intersect(r, &interaction);
      //SInteraction interaction = intersect(r);
      if (interaction.hitInformation.hit && r.m_medium) {
        *Tr *= r.m_medium->tr(r, sampler);
      }

      if (!interaction.hitInformation.hit || interaction.material) {
        return interaction;
      }

      p0 = interaction.hitInformation.pos;
      currentMedium = !r.m_medium ? interaction.medium : nullptr;
    }
  }

  inline void CDeviceScene::intersectTr(const CRay& ray, CSampler& sampler, glm::vec3* Tr, SInteraction* interaction) const {
    *Tr = glm::vec3(1.f);
    glm::vec3 p0 = ray.m_origin;
    const glm::vec3 p1 = p0 + ray.m_t_max * ray.m_direction;
    const CMedium* currentMedium = ray.m_medium;
    while (true) {
      CRay r = CRay::spawnRay(p0, p1, currentMedium);

      intersect(r, interaction);
      //SInteraction interaction = intersect(r);
      if (interaction->hitInformation.hit && r.m_medium) {
        *Tr *= r.m_medium->tr(r, sampler);
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