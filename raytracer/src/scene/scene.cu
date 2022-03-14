#include "scene/scene.hpp"
#include "scene/interaction.hpp"
#include "cuda_runtime.h"
#include "sampling/distribution_1d.hpp"
#include "scene/environmentmap.hpp"
#include <device_launch_parameters.h>
#include "utility/debugging.hpp"
#include "backend/rt_backend.hpp"
#include <iostream>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/mesh.h>
#include "backend/asset_manager.hpp"

namespace rt {

  CHostScene::CHostScene() :
    m_lightDist(nullptr),
    m_envMap(nullptr),
    m_traversableHandle(NULL),
    m_deviceIasBuffer(NULL),
    m_deviceInstances(NULL),
    m_hostDeviceConnection(this) {
  }

  CHostScene::~CHostScene() {
  }

  CHostScene::CHostScene(CHostScene&& scene):
    m_sceneobjects(std::move(scene.m_sceneobjects)),
    m_lights(std::move(scene.m_lights)),
    m_lightDist(std::exchange(scene.m_lightDist, nullptr)),
    m_envMap(std::exchange(scene.m_envMap, nullptr)),
    m_traversableHandle(std::move(scene.m_traversableHandle)),
    m_deviceIasBuffer(std::exchange(scene.m_deviceIasBuffer, NULL)),
    m_deviceInstances(std::exchange(scene.m_deviceInstances, NULL)),
    m_hostDeviceConnection(std::move(scene.m_hostDeviceConnection), this) {

  }

  CHostScene& CHostScene::operator=(CHostScene&& scene) {
    m_sceneobjects = std::move(scene.m_sceneobjects);
    m_lights = std::move(scene.m_lights);
    m_lightDist = std::exchange(scene.m_lightDist, nullptr);
    m_envMap = std::exchange(scene.m_envMap, nullptr);
    m_traversableHandle = std::move(scene.m_traversableHandle);
    m_deviceIasBuffer = std::exchange(scene.m_deviceIasBuffer, NULL);
    m_deviceInstances = std::exchange(scene.m_deviceInstances, NULL);
    m_hostDeviceConnection = std::move(scene.m_hostDeviceConnection);
    m_hostDeviceConnection.setHostScene(this);
    return *this;
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

  CSceneConnection::CSceneConnection(CSceneConnection&& connection, CHostScene* hostScene) :
    m_hostScene(hostScene),
    m_deviceScene(std::exchange(connection.m_deviceScene, nullptr)),
    m_deviceSceneobjects(std::exchange(connection.m_deviceSceneobjects, nullptr)),
    m_deviceLights(std::exchange(connection.m_deviceLights, nullptr)),
    m_deviceLightDist(std::exchange(connection.m_deviceLightDist, nullptr)),
    m_deviceEnvMap(std::exchange(connection.m_deviceEnvMap, nullptr)) {

  }
  CSceneConnection& CSceneConnection::operator=(CSceneConnection&& connection) {
    m_hostScene = std::exchange(connection.m_hostScene, nullptr);
    m_deviceScene = std::exchange(connection.m_deviceScene, nullptr);
    m_deviceSceneobjects = std::exchange(connection.m_deviceSceneobjects, nullptr);
    m_deviceLights = std::exchange(connection.m_deviceLights, nullptr);
    m_deviceLightDist = std::exchange(connection.m_deviceLightDist, nullptr);
    m_deviceEnvMap = std::exchange(connection.m_deviceEnvMap, nullptr);
    return *this;
  }

  void CSceneConnection::allocateDeviceMemory() {
    CUDA_ASSERT(cudaMalloc(&m_deviceScene, sizeof(CDeviceScene)));
    CUDA_ASSERT(cudaMalloc(&m_deviceSceneobjects, m_hostScene->m_sceneobjects.size() * sizeof(CDeviceSceneobject)));
    CUDA_ASSERT(cudaMalloc(&m_deviceLights, m_hostScene->m_lights.size() * sizeof(CDeviceSceneobject)));
    if (m_hostScene->m_envMap) {
      CUDA_ASSERT(cudaMalloc(&m_deviceEnvMap, sizeof(CEnvironmentMap)));
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
    CUDA_ASSERT(cudaMemcpy(m_deviceScene, &deviceScene, sizeof(CDeviceScene), cudaMemcpyHostToDevice));
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
    CUDA_ASSERT(cudaFree(reinterpret_cast<void*>(m_hostScene->m_deviceInstances)));
    CUDA_ASSERT(cudaFree(reinterpret_cast<void*>(m_hostScene->m_deviceIasBuffer)));

    for (auto& sceneObject : m_hostScene->m_sceneobjects) {
      sceneObject.freeDeviceMemory();
    }

    for (auto& light : m_hostScene->m_lights) {
      light.freeDeviceMemory();
    }

    if (m_hostScene->m_envMap) {
      m_hostScene->m_envMap->freeDeviceMemory();
    }

    CUDA_ASSERT(cudaFree(m_deviceLightDist));
    if (m_hostScene->m_lightDist) {
      m_hostScene->m_lightDist->freeDeviceMemory();
    }

    CUDA_ASSERT(cudaFree(m_deviceSceneobjects));
    CUDA_ASSERT(cudaFree(m_deviceScene));
  }

  

  //void CHostScene::createOptixProgramGroup() const {
  //  for (auto& sceneobject : m_sceneobjects) {
  //    sceneobject.createOptixProgramGroup();
  //  }
  //}

  void CHostScene::buildOptixAccel() {
    CAssetManager::buildOptixAccel(); // Build GAS

    // Build GAS for primitive sceneobjects and gather instances
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

  H_CALLABLE float roughnessFromExponent(float exponent) {
    return powf(2.f / (exponent + 2.f), 0.25f);
  }

  void CHostScene::addSceneobjectsFromAssimp(const std::string& assetsBasePath, const std::string& meshFileName, const glm::vec3& worldPos, const glm::vec3& normal, const glm::vec3& scaling, ESceneobjectMask mask) {
    std::vector<std::tuple<CMesh*, CMaterial*>> meshdata = CAssetManager::loadMesh(assetsBasePath, meshFileName);

    for (auto m : meshdata) {
      auto[mesh, material] = m;
      addSceneobject(CHostSceneobject(mesh, material, worldPos, normal, scaling, mask));
    }

  }

  std::tuple<std::vector<SAABB>, std::vector<SAABB>, glm::mat4x3> CHostScene::getObjectBBs(ESceneobjectMask mask) const {
    std::string objectPath = "";
    std::vector<SAABB> modelBBs;
    std::vector<SAABB> worldBBs;
    glm::mat4x3 worldToModel;
    for (const auto& sceneobject : m_sceneobjects) {
      if (sceneobject.mask() & mask) {
        if (!sceneobject.mesh()) {
          throw std::runtime_error("No mesh available");
        }
        if (objectPath == "") {
          objectPath = sceneobject.mesh()->path();
        }
        if (objectPath != sceneobject.mesh()->path()) {
          throw std::runtime_error("Filtering only available for single mesh");
        }
        modelBBs.push_back(sceneobject.modelAABB());
        worldBBs.push_back(sceneobject.worldAABB());
        worldToModel = sceneobject.worldToModel();
      }
    }
    return { modelBBs, worldBBs, worldToModel};
  }
}