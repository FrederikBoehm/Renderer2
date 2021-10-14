#include "scene/scene.hpp"
#include "scene/interaction.hpp"
#include "cuda_runtime.h"
#include "sampling/distribution_1d.hpp"
#include "scene/environmentmap.hpp"

namespace rt {

  CHostScene::CHostScene() :
    m_lightDist(nullptr),
    m_envMap(nullptr),
    m_hostDeviceConnection(this) {

  }

  void CHostScene::addSceneobject(CHostSceneobject&& sceneobject) {
    m_sceneobjects.push_back(std::move(sceneobject));
    
  }

  void CHostScene::addLightsource(CHostSceneobject&& lightsource) {
    m_lights.push_back(std::move(lightsource));
    if (m_lightDist) {
      delete m_lightDist;
    }
    std::vector<float> powers;
    powers.reserve(m_lights.size());
    for (auto& lightsource : m_lights) {
      powers.push_back(lightsource.power());
    }
    m_lightDist = new CDistribution1D(powers);
  }

  void CHostScene::setEnvironmentMap(CEnvironmentMap&& envMap) {
    if (!m_envMap) {
      m_envMap = new CEnvironmentMap(std::move(envMap));
    }
    else {
      *m_envMap = std::move(envMap);
    }
  }

  CSceneConnection::CSceneConnection(CHostScene* hostScene) :
    m_hostScene(hostScene),
    m_deviceScene(nullptr),
    m_deviceSceneobjects(nullptr),
    m_deviceLights(nullptr),
    m_deviceLightDist(nullptr),
    m_deviceEnvMap(nullptr) {

  }

  void CSceneConnection::allocateDeviceMemory() {
    cudaMalloc(&m_deviceScene, sizeof(CDeviceScene));
    cudaMalloc(&m_deviceSceneobjects, m_hostScene->m_sceneobjects.size() * sizeof(CDeviceSceneobject));
    cudaMalloc(&m_deviceLights, m_hostScene->m_lights.size() * sizeof(CDeviceSceneobject));
    //cudaMalloc(&m_deviceLightDist, sizeof(CDistribution1D));

    if (m_hostScene->m_envMap) {
      cudaMalloc(&m_deviceEnvMap, sizeof(CEnvironmentMap));
      m_hostScene->m_envMap->allocateDeviceMemory();
    }

    for (auto& sceneObject : m_hostScene->m_sceneobjects) {
      sceneObject.allocateDeviceMemory();
    }

    for (auto& light : m_hostScene->m_lights) {
      light.allocateDeviceMemory();
    }
  }

  void CSceneConnection::copyToDevice() {
    CDeviceScene deviceScene;
    deviceScene.m_numSceneobjects = m_hostScene->m_sceneobjects.size();
    deviceScene.m_sceneobjects = m_deviceSceneobjects;
    deviceScene.m_numLights = m_hostScene->m_lights.size();
    deviceScene.m_lights = m_deviceLights;
    deviceScene.m_lightDist = m_deviceLightDist;
    deviceScene.m_envMap = m_deviceEnvMap;
    cudaMemcpy(m_deviceScene, &deviceScene, sizeof(CDeviceScene), cudaMemcpyHostToDevice);
    for (size_t i = 0; i < m_hostScene->m_sceneobjects.size(); ++i) {
      m_hostScene->m_sceneobjects[i].setDeviceSceneobject(&m_deviceSceneobjects[i]);
      m_hostScene->m_sceneobjects[i].copyToDevice();
    }
    for (size_t i = 0; i < m_hostScene->m_lights.size(); ++i) {
      m_hostScene->m_lights[i].setDeviceSceneobject(&m_deviceLights[i]);
      m_hostScene->m_lights[i].copyToDevice();
    }
    //cudaMemcpy(m_deviceLightDist, m_hostScene->m_lightDist, sizeof(CDistribution1D), cudaMemcpyHostToDevice);
    //m_hostScene->m_lightDist->copyToDevice(m_deviceLightDist);
    if (m_hostScene->m_envMap) {
      m_hostScene->m_envMap->copyToDevice(m_deviceEnvMap);
    }
  }

  void CSceneConnection::freeDeviceMemory() {
    for (auto& sceneObject : m_hostScene->m_sceneobjects) {
      sceneObject.freeDeviceMemory();
    }

    for (auto& light : m_hostScene->m_lights) {
      light.freeDeviceMemory();
    }

    if (m_hostScene->m_envMap) {
      m_hostScene->m_envMap->freeDeviceMemory();
    }

    cudaFree(m_deviceLightDist);
    m_hostScene->m_lightDist->freeDeviceMemory();

    cudaFree(m_deviceSceneobjects);
    cudaFree(m_deviceScene);
  }

  SInteraction CDeviceScene::intersect(const CRay& ray) const {
    SInteraction closestInteraction;
    closestInteraction.hitInformation.hit = false;
    closestInteraction.hitInformation.t = 1e+10;
    closestInteraction.object = nullptr;
    closestInteraction.material = nullptr;
    closestInteraction.medium = nullptr;
    CDeviceSceneobject* sceneobjects = m_sceneobjects;
    for (uint8_t i = 0; i < m_numSceneobjects; ++i) {
      SInteraction currentInteraction = m_sceneobjects[i].intersect(ray);
      if (currentInteraction.hitInformation.hit && currentInteraction.hitInformation.t < closestInteraction.hitInformation.t) {
        closestInteraction = currentInteraction;
      }
    }
    for (uint8_t i = 0; i < m_numLights; ++i) {
      SInteraction currentInteraction = m_lights[i].intersect(ray);
      if (currentInteraction.hitInformation.hit && currentInteraction.hitInformation.t < closestInteraction.hitInformation.t) {
        closestInteraction = currentInteraction;
      }
    }
    if (closestInteraction.hitInformation.hit) {
      ray.m_t_max = closestInteraction.hitInformation.t;
    }
    return closestInteraction;
  }

  //glm::vec3 CDeviceScene::sampleLightSources(CSampler& sampler, float* pdf) const {
  //  size_t index = m_lightDist->sampleDiscrete(sampler, pdf);
  //  const CShape* lightGeometry = m_lights[index].shape();
  //  switch (lightGeometry->shape()) {
  //  case EShape::PLANE:
  //    return ((const Plane*)lightGeometry)->sample(sampler);
  //  }
  //  return glm::vec3(0.0f);
  //}

  glm::vec3 CDeviceScene::sampleLightSources(CSampler& sampler, glm::vec3* direction, float* pdf) const {
    if (m_envMap) {
      return m_envMap->sample(sampler, direction, pdf);
    }
    return glm::vec3(0.f);
  }

  glm::vec3 CDeviceScene::le(const glm::vec3& direction, float* pdf) const {
    if (m_envMap) {
      return m_envMap->le(direction, pdf);
    }
    return glm::vec3(0.f);
  }

  float CDeviceScene::lightSourcesPdf(const SInteraction& lightHit) const {
    if (lightHit.object) {
      float power = lightHit.object->power();
      float totalPower = m_lightDist->integral();
      return power / totalPower;
    }
    return 0.0f;
  }

  float CDeviceScene::lightSourcePdf(const SInteraction& lightHit, const CRay& shadowRay) const {
    if (lightHit.object) {
      const CShape* lightGeometry = lightHit.object->shape();
      switch (lightGeometry->shape()) {
      case EShape::PLANE: {

        return ((const Plane*)lightGeometry)->pdf(lightHit, shadowRay);
      }
      }
    }
    return 0.0f;
  }

  bool CDeviceScene::occluded(const CRay& ray) const {
    return intersect(ray).hitInformation.hit;
  }

  glm::vec3 CDeviceScene::tr(const CRay& ray, CSampler& sampler) const {
    glm::vec3 p0 = ray.m_origin;
    const glm::vec3 p1 = p0 + ray.m_t_max * ray.m_direction;
    glm::vec3 Tr(1.f);
    while (true) {
      CRay r = CRay::spawnRay(p0, p1);
      SInteraction interaction = intersect(r);
      if (interaction.hitInformation.hit && interaction.material) {
        return glm::vec3(0.f);
      }
      if (interaction.medium) {
        Tr *= interaction.medium->tr(r, sampler);
      }

      if (!interaction.hitInformation.hit) {
        break;
      }

      p0 = interaction.hitInformation.pos;
    }
    return Tr;
  }

  SInteraction CDeviceScene::intersectTr(const CRay& ray, CSampler& sampler, glm::vec3* Tr) const {
    *Tr = glm::vec3(1.f);
    glm::vec3 p0 = ray.m_origin;
    const glm::vec3 p1 = p0 + ray.m_t_max * ray.m_direction;
    while (true) {
      CRay r = CRay::spawnRay(p0, p1);

      SInteraction interaction = intersect(r);
      if (interaction.medium) {
        *Tr *= interaction.medium->tr(r, sampler);
      }

      //bool one = !interaction.hitInformation.hit;
      //bool two = interaction.material;
      //bool eval = one || two;
      if (!interaction.hitInformation.hit || interaction.material) {
      //if (interaction.material) {
      //if (eval) {
        return interaction;
      }

      //if (!eval) {
      //  bool v = false;
      //}
      //else {
      //  return interaction;
      //}

      p0 = interaction.hitInformation.pos;
    }
  }
}