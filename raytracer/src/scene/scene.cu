#include "scene/scene.hpp"
#include "scene/interaction.hpp"
#include "cuda_runtime.h"
#include "sampling/distribution_1d.hpp"
#include "scene/environmentmap.hpp"
#include <device_launch_parameters.h>
#include "utility/debugging.hpp"
#include "backend/rt_backend.hpp"

namespace rt {

  CHostScene::CHostScene() :
    m_lightDist(nullptr),
    m_envMap(nullptr),
    m_hostDeviceConnection(this),
    m_deviceIasBuffer(NULL) {
  }

  CHostScene::~CHostScene() {
#ifndef __CUDA_ARCH__
    cudaFree((void*)m_deviceIasBuffer);
    cudaFree(reinterpret_cast<void*>(m_deviceInstances));
#endif
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

    if (m_hostScene->m_envMap) {
      cudaMalloc(&m_deviceEnvMap, sizeof(CEnvironmentMap));
      m_hostScene->m_envMap->allocateDeviceMemory();
    }

    //for (auto& sceneObject : m_hostScene->m_sceneobjects) {
    for (size_t i = 0; i < m_hostScene->m_sceneobjects.size(); ++i) {
      m_hostScene->m_sceneobjects[i].allocateDeviceMemory();
      m_hostScene->m_sceneobjects[i].setDeviceSceneobject(&m_deviceSceneobjects[i]);
    }

    //for (auto& light : m_hostScene->m_lights) {
    for (size_t i = 0; i < m_hostScene->m_lights.size(); ++i) {
      m_hostScene->m_lights[i].allocateDeviceMemory();
      m_hostScene->m_lights[i].setDeviceSceneobject(&m_deviceLights[i]);
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
    deviceScene.m_traversableHandle = m_hostScene->m_traversableHandle;
    cudaMemcpy(m_deviceScene, &deviceScene, sizeof(CDeviceScene), cudaMemcpyHostToDevice);
    for (size_t i = 0; i < m_hostScene->m_sceneobjects.size(); ++i) {
      //m_hostScene->m_sceneobjects[i].setDeviceSceneobject(&m_deviceSceneobjects[i]);
      m_hostScene->m_sceneobjects[i].copyToDevice();
    }
    for (size_t i = 0; i < m_hostScene->m_lights.size(); ++i) {
      //m_hostScene->m_lights[i].setDeviceSceneobject(&m_deviceLights[i]);
      m_hostScene->m_lights[i].copyToDevice();
    }
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
    if (m_hostScene->m_lightDist) {
      m_hostScene->m_lightDist->freeDeviceMemory();
    }

    cudaFree(m_deviceSceneobjects);
    cudaFree(m_deviceScene);
  }

  

  //void CHostScene::createOptixProgramGroup() const {
  //  for (auto& sceneobject : m_sceneobjects) {
  //    sceneobject.createOptixProgramGroup();
  //  }
  //}

  void CHostScene::buildOptixAccel() {
    // Build GAS for sceneobjects and gather instances
    std::vector<OptixInstance> instances;
    instances.reserve(m_sceneobjects.size());
    uint32_t instanceId = 0;
    uint32_t sbtOffset = 0;
    for (auto& sceneobject : m_sceneobjects) {
      sceneobject.buildOptixAccel();
      instances.push_back(sceneobject.getOptixInstance(instanceId, sbtOffset));
      ++instanceId;
      sbtOffset += RAY_TYPE_COUNT;
    }

    size_t instancesBytes = sizeof(OptixInstance) * instances.size();
    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void**>(&m_deviceInstances), instancesBytes));
    CUDA_ASSERT(cudaMemcpy(reinterpret_cast<void*>(m_deviceInstances), instances.data(), instancesBytes, cudaMemcpyHostToDevice));

    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    buildInput.instanceArray.instances = m_deviceInstances;
    buildInput.instanceArray.numInstances = instances.size();

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    CRTBackend* backend = CRTBackend::instance();

    OptixAccelBufferSizes bufferSizes;
    OPTIX_ASSERT(optixAccelComputeMemoryUsage(
      backend->context(),
      &accelOptions,
      &buildInput,
      1, // num build inputs
      &bufferSizes
    ));

    CUdeviceptr d_tempBuffer;
    CUDA_ASSERT(cudaMalloc(
      reinterpret_cast<void**>(&d_tempBuffer),
      bufferSizes.tempSizeInBytes
    ));
    CUDA_ASSERT(cudaMalloc(
      reinterpret_cast<void**>(&m_deviceIasBuffer),
      bufferSizes.outputSizeInBytes
    ));

    OPTIX_ASSERT(optixAccelBuild(
      backend->context(),
      nullptr,                  // CUDA stream
      &accelOptions,
      &buildInput,
      1,                  // num build inputs
      d_tempBuffer,
      bufferSizes.tempSizeInBytes,
      m_deviceIasBuffer,
      bufferSizes.outputSizeInBytes,
      &m_traversableHandle,
      nullptr,            // emitted property list
      0                   // num emitted properties
    ));

    CUDA_ASSERT(cudaFree(reinterpret_cast<void*>(d_tempBuffer)));

    //// make update temp buffer for ias
    //CUDA_ASSERT(cudaMalloc(reinterpret_cast<void**>(&ias.d_update_buffer),
    //  ias.buffer_sizes.tempUpdateSizeInBytes));
  }

  std::vector<SRecord<const CDeviceSceneobject*>> CHostScene::getSBTHitRecords() const {
    std::vector<SRecord<const CDeviceSceneobject*>> sbtHitRecords;
    sbtHitRecords.reserve(m_sceneobjects.size());
    for (const auto& sceneobject : m_sceneobjects) {
      sbtHitRecords.push_back(sceneobject.getSBTHitRecord());
    }
    return sbtHitRecords;
  }
}