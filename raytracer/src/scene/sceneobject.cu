#include "scene/sceneobject.hpp"
#include <iostream>

#include "shapes/circle.hpp"
#include "shapes/sphere.hpp"
#include "medium/homogeneous_medium.hpp"
#include "shapes/rectangle.hpp"
#include "shapes/cuboid.hpp"
#include "medium/heterogenous_medium.hpp"
#include "medium/nvdb_medium.hpp"
#include "backend/rt_backend.hpp"
#include "utility/debugging.hpp"
#include <optix/optix_stubs.h>
#include "utility/functions.hpp"
#include "backend/asset_manager.hpp"

namespace rt {
  std::shared_ptr<CShape> CHostSceneobject::getShape(EShape shape, const glm::vec3& worldPos, float radius, const glm::vec3& normal) {
    switch (shape) {
    case EShape::CIRCLE:
      return std::make_shared<CCircle>(worldPos, radius, normal);
      break;
    case EShape::SPHERE:
      return std::make_shared<Sphere>(worldPos, radius, normal);
    }
  }

  CHostSceneobject::CHostSceneobject(CShape* shape, const glm::vec3& diffuseReflection, float diffuseRougness, const glm::vec3& specularReflection, float alphaX, float alphaY, float etaI, float etaT) :
    m_shape(shape),
    m_mesh(nullptr),
    m_material(nullptr),
    m_medium(nullptr),
    m_flag(ESceneobjectFlag::GEOMETRY),
    m_deviceGasBuffer(NULL),
    m_hostDeviceConnection(this) {
    m_material = new CMaterial(diffuseReflection, specularReflection, COrenNayarBRDF(diffuseRougness), CMicrofacetBRDF(alphaX, alphaY, etaI, etaT));

  }

  CHostSceneobject::CHostSceneobject(CNVDBMedium* medium, const glm::vec3& worldPos, const glm::vec3& orientation, const glm::vec3& scaling) :
    m_shape(nullptr),
    m_mesh(nullptr),
    m_material(nullptr),
    m_medium(new CMediumInstance(medium, &m_modelToWorld, &m_worldToModel)),
    m_flag(ESceneobjectFlag::VOLUME),
    m_modelToWorld(getModelToWorldTransform(worldPos, orientation, scaling)),
    m_worldToModel(glm::inverse(glm::mat4(m_modelToWorld))),
    m_deviceGasBuffer(NULL),
    m_hostDeviceConnection(this) {
  }

  CHostSceneobject::CHostSceneobject(CHostSceneobject&& sceneobject) :
    m_shape(std::move(sceneobject.m_shape)),
    m_mesh(std::exchange(sceneobject.m_mesh, nullptr)),
    m_material(std::exchange(sceneobject.m_material, nullptr)),
    m_medium(std::move(sceneobject.m_medium)),
    m_flag(std::move(sceneobject.m_flag)),
    m_modelToWorld(std::move(sceneobject.m_modelToWorld)),
    m_worldToModel(std::move(sceneobject.m_worldToModel)),
    m_deviceGasBuffer(std::exchange(sceneobject.m_deviceGasBuffer, NULL)),
    m_hostDeviceConnection(this) {
  }

  CHostSceneobject::CHostSceneobject(CMesh* mesh, CMaterial* material):
    m_shape(nullptr),
    m_mesh(mesh),
    m_material(material),
    m_medium(nullptr),
    m_flag(ESceneobjectFlag::GEOMETRY),
    m_deviceGasBuffer(NULL),
    m_hostDeviceConnection(this) {

  }

  CHostSceneobject::CHostSceneobject(CMesh* mesh, CMaterial* material, const glm::vec3& worldPos, const glm::vec3& orientation, const glm::vec3& scaling) :
    m_shape(nullptr),
    m_mesh(mesh),
    m_material(material),
    m_medium(nullptr),
    m_flag(ESceneobjectFlag::GEOMETRY),
    m_modelToWorld(getModelToWorldTransform(worldPos, orientation, scaling)),
    m_worldToModel(glm::inverse(glm::mat4(m_modelToWorld))),
    m_deviceGasBuffer(NULL),
    m_hostDeviceConnection(this) {

  }

  CSceneobjectConnection::CSceneobjectConnection(CHostSceneobject* hostSceneobject):
    m_hostSceneobject(hostSceneobject) {
  }

  CSceneobjectConnection::CSceneobjectConnection(const CSceneobjectConnection&& connection) :
    m_hostSceneobject(std::move(connection.m_hostSceneobject)) {
  }

  void CSceneobjectConnection::allocateDeviceMemory() {
    if (m_hostSceneobject->m_shape) {
      switch (m_hostSceneobject->m_shape->shape()) {
      case EShape::CIRCLE:
        CUDA_ASSERT(cudaMalloc(&m_deviceShape, sizeof(CCircle)));
        break;
      case EShape::SPHERE:
        CUDA_ASSERT(cudaMalloc(&m_deviceShape, sizeof(Sphere)));
        break;
      case EShape::RECTANGLE:
        CUDA_ASSERT(cudaMalloc(&m_deviceShape, sizeof(CRectangle)));
        break;
      case EShape::CUBOID:
        CUDA_ASSERT(cudaMalloc(&m_deviceShape, sizeof(CCuboid)));
        break;
      }
    }
    if (m_hostSceneobject->m_material && m_hostSceneobject->m_shape) {
      CUDA_ASSERT(cudaMalloc(&m_deviceMaterial, sizeof(CMaterial)));
      m_hostSceneobject->m_material->allocateDeviceMemory();
    }
    if (m_hostSceneobject->m_medium) {
      CUDA_ASSERT(cudaMalloc(&m_deviceMedium, sizeof(CMediumInstance)));
    }
    
  }

  __global__ void getTransformPointers(CDeviceSceneobject* sceneobject, glm::mat4x3** modelToWorld, glm::mat4x3** worldToModel) {
    *modelToWorld = &sceneobject->m_modelToWorld;
    *worldToModel = &sceneobject->m_worldToModel;
  }

  void CSceneobjectConnection::copyToDevice() {
    if (m_deviceShape) {
      switch (m_hostSceneobject->m_shape->shape()) {
      case EShape::CIRCLE:
        CUDA_ASSERT(cudaMemcpy(m_deviceShape, m_hostSceneobject->m_shape.get(), sizeof(CCircle), cudaMemcpyHostToDevice));
        break;
      case EShape::SPHERE:
        CUDA_ASSERT(cudaMemcpy(m_deviceShape, m_hostSceneobject->m_shape.get(), sizeof(Sphere), cudaMemcpyHostToDevice));
        break;
      case EShape::RECTANGLE:
        CUDA_ASSERT(cudaMemcpy(m_deviceShape, m_hostSceneobject->m_shape.get(), sizeof(CRectangle), cudaMemcpyHostToDevice));
        break;
      case EShape::CUBOID:
        CUDA_ASSERT(cudaMemcpy(m_deviceShape, m_hostSceneobject->m_shape.get(), sizeof(CCuboid), cudaMemcpyHostToDevice));
        break;
      }
    }
    if (m_deviceMaterial && m_deviceShape) {
      CUDA_ASSERT(cudaMemcpy(m_deviceMaterial, &m_hostSceneobject->m_material->copyToDevice(), sizeof(CMaterial), cudaMemcpyHostToDevice));
    }
    if (m_deviceMedium) {
      // Get device pointers for sceneobject transforms
      glm::mat4x3** d_modelToWorld;
      glm::mat4x3** d_worldToModel;
      CUDA_ASSERT(cudaMalloc(&d_modelToWorld, sizeof(glm::mat4x3*)));
      CUDA_ASSERT(cudaMalloc(&d_worldToModel, sizeof(glm::mat4x3*)));

      getTransformPointers << <1, 1 >> > (m_deviceSceneobject, d_modelToWorld, d_worldToModel);
      CUDA_ASSERT(cudaDeviceSynchronize());

      glm::mat4x3* modelToWorldPtr;
      glm::mat4x3* worldToModelPtr;
      CUDA_ASSERT(cudaMemcpy(&modelToWorldPtr, d_modelToWorld, sizeof(glm::mat4x3*), cudaMemcpyDeviceToHost));
      CUDA_ASSERT(cudaMemcpy(&worldToModelPtr, d_worldToModel, sizeof(glm::mat4x3*), cudaMemcpyDeviceToHost));

      CMediumInstance deviceMedium(CAssetManager::deviceMedium(m_hostSceneobject->m_medium->path()), modelToWorldPtr, worldToModelPtr);
      CUDA_ASSERT(cudaMemcpy(m_deviceMedium, &deviceMedium, sizeof(CMediumInstance), cudaMemcpyHostToDevice));

    }
    if (m_deviceSceneobject) {

      CDeviceSceneobject deviceSceneobject;
      deviceSceneobject.m_shape = m_deviceShape;
      deviceSceneobject.m_mesh = m_hostSceneobject->m_mesh ? CAssetManager::deviceMesh(m_hostSceneobject->m_mesh->path(), m_hostSceneobject->m_mesh->submeshId()) : nullptr;
      if (m_hostSceneobject->m_material) {
        if (m_hostSceneobject->m_shape) {
          deviceSceneobject.m_material = m_deviceMaterial;
        }
        else {
          deviceSceneobject.m_material = CAssetManager::deviceMaterial(m_hostSceneobject->m_material->path(), m_hostSceneobject->m_material->submeshId());
        }
      }
      else {
        deviceSceneobject.m_material = nullptr;
      }
      deviceSceneobject.m_medium = m_deviceMedium;
      deviceSceneobject.m_flag = m_hostSceneobject->m_flag;
      deviceSceneobject.m_modelToWorld = m_hostSceneobject->m_modelToWorld;
      deviceSceneobject.m_worldToModel = m_hostSceneobject->m_worldToModel;
      CUDA_ASSERT(cudaMemcpy(m_deviceSceneobject, &deviceSceneobject, sizeof(CDeviceSceneobject), cudaMemcpyHostToDevice));
    }
  }

  void CSceneobjectConnection::freeDeviceMemory() {
    CUDA_ASSERT(cudaFree(reinterpret_cast<void*>(m_hostSceneobject->m_deviceGasBuffer)));
    m_hostSceneobject->m_deviceGasBuffer = NULL;
    CUDA_ASSERT(cudaFree(m_deviceShape));
    m_deviceShape = nullptr;
    if (m_deviceMaterial && m_hostSceneobject->m_shape) {
      m_hostSceneobject->m_material->freeDeviceMemory();
      CUDA_ASSERT(cudaFree(m_deviceMaterial));
      m_deviceMaterial = nullptr;
    }
  }

  float CHostSceneobject::power() const {
    if (m_flag == ESceneobjectFlag::GEOMETRY) {
      glm::vec3 L = m_material->Le();
      switch (m_shape->shape()) {
      case EShape::CIRCLE:
        return (L.x + L.y + L.z) * ((CCircle*)m_shape.get())->area();
      }
    }
    return 0.0f;
  }

  CHostSceneobject::~CHostSceneobject() {
    if (m_shape) { // TODO: Find better way to clean up material if using primitive shape
      delete m_material;
    }
  }

  void CHostSceneobject::buildOptixAccel() {
    SBuildInputWrapper buildInputWrapper;
    if (m_shape) {
      buildInputWrapper = m_shape->getOptixBuildInput();
    }
    else if (m_medium || m_mesh) {
      return;
    }
    else {
      fprintf(stderr, "[ERROR] Could not create OptiX accel.\n");
      return;
    }

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gasBufferSizes;
    const OptixDeviceContext& context = CRTBackend::instance()->context();
    OPTIX_ASSERT(optixAccelComputeMemoryUsage(context, &accelOptions, &buildInputWrapper.buildInput, 1, &gasBufferSizes));

    CUdeviceptr d_tempBufferGas;
    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void**>(&d_tempBufferGas), gasBufferSizes.tempSizeInBytes));
    CUdeviceptr d_outputBufferGas;
    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void**>(&d_outputBufferGas), gasBufferSizes.outputSizeInBytes));
    CUdeviceptr d_compactedSize;
    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void**>(&d_compactedSize), sizeof(size_t)));

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = d_compactedSize;


    OPTIX_ASSERT(optixAccelBuild(CRTBackend::instance()->context(),
                                 0,
                                 &accelOptions,
                                 &buildInputWrapper.buildInput,
                                 1,
                                 d_tempBufferGas,
                                 gasBufferSizes.tempSizeInBytes,
                                 d_outputBufferGas,
                                 gasBufferSizes.outputSizeInBytes,
                                 &m_traversableHandle,
                                 &emitProperty,
                                 1));

    CUDA_ASSERT(cudaStreamSynchronize(0));

    size_t compactedSize;
    CUDA_ASSERT(cudaMemcpy(&compactedSize, reinterpret_cast<void*>(emitProperty.result), sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaFree(reinterpret_cast<void*>(d_compactedSize)));
    if (compactedSize < gasBufferSizes.outputSizeInBytes)
    {
      CUDA_ASSERT(cudaMalloc(reinterpret_cast<void**>(&m_deviceGasBuffer), compactedSize));
      OPTIX_ASSERT(optixAccelCompact(context, 0, m_traversableHandle, m_deviceGasBuffer, compactedSize, &m_traversableHandle));
      CUDA_ASSERT(cudaFree(reinterpret_cast<void*>(d_outputBufferGas)));
    }
    else
    {
      m_deviceGasBuffer = d_outputBufferGas;
    }
    CUDA_ASSERT(cudaFree(reinterpret_cast<void*>(d_tempBufferGas)));
  }

  void CHostSceneobject::getTransform(float* outMatrix) const {
    if (m_mesh || m_medium) {
      const glm::mat4x3& modelToWorldTransform = m_modelToWorld;
      outMatrix[0]    = modelToWorldTransform[0][0]; // Column to row major
      outMatrix[1]    = modelToWorldTransform[1][0];
      outMatrix[2]    = modelToWorldTransform[2][0];
      outMatrix[3]    = modelToWorldTransform[3][0];
      outMatrix[4]    = modelToWorldTransform[0][1];
      outMatrix[5]    = modelToWorldTransform[1][1];
      outMatrix[6]    = modelToWorldTransform[2][1];
      outMatrix[7]    = modelToWorldTransform[3][1];
      outMatrix[8]    = modelToWorldTransform[0][2];
      outMatrix[9]    = modelToWorldTransform[1][2];
      outMatrix[10]   = modelToWorldTransform[2][2];
      outMatrix[11]   = modelToWorldTransform[3][2];
    }
    else {
      float identity[] = { 1.f, 0.f, 0.f, 0.f,
                           0.f, 1.f, 0.f, 0.f,
                           0.f, 0.f, 1.f, 0.f };
      memcpy(outMatrix, identity, sizeof(float) * 12);
    }
  }

  OptixInstance CHostSceneobject::getOptixInstance(uint32_t instanceId, uint32_t sbtOffset) const {
    OptixInstance instance;

    instance.flags = OPTIX_INSTANCE_FLAG_NONE;
    instance.instanceId = instanceId;
    instance.sbtOffset = sbtOffset;
    instance.visibilityMask = 0xff; // TODO: Check what has to be set here
    if (m_mesh) {
      instance.traversableHandle = m_mesh->getOptixHandle();
    }
    else if (m_medium) {
      instance.traversableHandle = m_medium->getOptixHandle();
    }
    else {
      instance.traversableHandle = m_traversableHandle;
    }
    getTransform(instance.transform);

    return instance;
  }

  OptixProgramGroup CHostSceneobject::getOptixProgramGroup() const {
    if (m_medium.get()) {
      return m_medium->getOptixProgramGroup();
    }
    else if (m_shape.get()) {
      return m_shape->getOptixProgramGroup();
    }
    else if (m_mesh) {
      return m_mesh->getOptixProgramGroup();
    }
    fprintf(stderr, "[ERROR] CHostSceneobject::getOptixProgramGroup no valid program group found.\n");
    return OptixProgramGroup();
  }

  SRecord<const CDeviceSceneobject*> CHostSceneobject::getSBTHitRecord() const {
    SRecord<const CDeviceSceneobject*> hitRecord;
    OPTIX_ASSERT(optixSbtRecordPackHeader(getOptixProgramGroup(), &hitRecord));
    if (!m_hostDeviceConnection.deviceSceneobject()) {
      fprintf(stderr, "[ERROR] CHostSceneobject::getSBTHitRecord: deviceSceneobject is null.\n");
    }
    hitRecord.data = m_hostDeviceConnection.deviceSceneobject();
    return hitRecord;
  }

  glm::mat4 CHostSceneobject::getModelToWorldTransform(const glm::vec3& worldPos, const glm::vec3& orientation, const glm::vec3& scaling) {
    return glm::translate(worldPos) * getRotation(orientation) * glm::scale(scaling);
  }


}