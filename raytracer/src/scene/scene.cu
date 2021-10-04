#include "scene/scene.hpp"
#include "scene/surface_interaction.hpp"
#include "cuda_runtime.h"
#include "sampling/distribution_1d.hpp"
#include "scene/surface_interaction.hpp"

namespace rt {

  CHostScene::CHostScene() :
    m_lightDist(nullptr),
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

  CSceneConnection::CSceneConnection(CHostScene* hostScene) :
    m_hostScene(hostScene) {

  }

  void CSceneConnection::allocateDeviceMemory() {
    cudaMalloc(&m_deviceScene, sizeof(CDeviceScene));
    cudaMalloc(&m_deviceSceneobjects, m_hostScene->m_sceneobjects.size() * sizeof(CDeviceSceneobject));
    cudaMalloc(&m_deviceLights, m_hostScene->m_lights.size() * sizeof(CDeviceSceneobject));
    cudaMalloc(&m_deviceLightDist, sizeof(CDistribution1D));

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
    cudaMemcpy(m_deviceScene, &deviceScene, sizeof(CDeviceScene), cudaMemcpyHostToDevice);
    for (size_t i = 0; i < m_hostScene->m_sceneobjects.size(); ++i) {
      m_hostScene->m_sceneobjects[i].setDeviceSceneobject(&m_deviceSceneobjects[i]);
      m_hostScene->m_sceneobjects[i].copyToDevice();
    }
    for (size_t i = 0; i < m_hostScene->m_lights.size(); ++i) {
      m_hostScene->m_lights[i].setDeviceSceneobject(&m_deviceLights[i]);
      m_hostScene->m_lights[i].copyToDevice();
    }
    cudaMemcpy(m_deviceLightDist, m_hostScene->m_lightDist, sizeof(CDistribution1D), cudaMemcpyHostToDevice);
    m_hostScene->m_lightDist->copyToDevice(m_deviceLightDist);
  }

  void CSceneConnection::freeDeviceMemory() {
    for (auto& sceneObject : m_hostScene->m_sceneobjects) {
      sceneObject.freeDeviceMemory();
    }

    for (auto& light : m_hostScene->m_lights) {
      light.freeDeviceMemory();
    }

    cudaFree(m_deviceLightDist);
    m_hostScene->m_lightDist->freeDeviceMemory();

    cudaFree(m_deviceSceneobjects);
    cudaFree(m_deviceScene);
  }

  SSurfaceInteraction CDeviceScene::intersect(const Ray& ray) const {
    SSurfaceInteraction closestInteraction;
    closestInteraction.hitInformation.t = 1e+10;
    closestInteraction.object = nullptr;
    CDeviceSceneobject* sceneobjects = m_sceneobjects;
    for (uint8_t i = 0; i < m_numSceneobjects; ++i) {
      SSurfaceInteraction currentInteraction = m_sceneobjects[i].intersect(ray);
      if (currentInteraction.hitInformation.hit && currentInteraction.hitInformation.t < closestInteraction.hitInformation.t) {
        closestInteraction = currentInteraction;
      }
    }
    for (uint8_t i = 0; i < m_numLights; ++i) {
      SSurfaceInteraction currentInteraction = m_lights[i].intersect(ray);
      if (currentInteraction.hitInformation.hit && currentInteraction.hitInformation.t < closestInteraction.hitInformation.t) {
        closestInteraction = currentInteraction;
      }
    }
    return closestInteraction;
  }

  glm::vec3 CDeviceScene::sampleLightSources(CSampler& sampler, float* pdf) const {
    size_t index = m_lightDist->sampleDiscrete(sampler, pdf);
    const CShape* lightGeometry = m_lights[index].shape();
    switch (lightGeometry->shape()) {
    case EShape::PLANE:
      return ((const Plane*)lightGeometry)->sample(sampler);
    }
    return glm::vec3(0.0f);
  }

  float CDeviceScene::lightSourcesPdf(const SSurfaceInteraction& lightHit) const {
    if (lightHit.object) {
      float power = lightHit.object->power();
      float totalPower = m_lightDist->integral();
      return power / totalPower;
    }
    return 1.0f;
  }

  float CDeviceScene::lightSourcePdf(const SSurfaceInteraction& lightHit, const Ray& shadowRay) const {
    if (lightHit.object) {
      const CShape* lightGeometry = lightHit.object->shape();
      switch (lightGeometry->shape()) {
      case EShape::PLANE: {

        return ((const Plane*)lightGeometry)->pdf(lightHit, shadowRay);
      }
      }
    }
    return 1.0f;
  }
}