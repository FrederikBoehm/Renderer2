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

  CHostSceneobject::CHostSceneobject(CShape* shape, const glm::vec3& le):
    m_shape(shape),
    m_mesh(nullptr),
    m_material(nullptr),
    m_medium(nullptr),
    m_flag(ESceneobjectFlag::GEOMETRY),
    m_deviceGasBuffer(NULL),
    m_hostDeviceConnection(this) {
    m_material = std::make_shared<CMaterial>(CMaterial(le));
  }

  CHostSceneobject::CHostSceneobject(CShape* shape, const glm::vec3& diffuseReflection, float diffuseRougness, const glm::vec3& specularReflection, float alphaX, float alphaY, float etaI, float etaT) :
    m_shape(shape),
    m_mesh(nullptr),
    m_material(nullptr),
    m_medium(nullptr),
    m_flag(ESceneobjectFlag::GEOMETRY),
    m_deviceGasBuffer(NULL),
    m_hostDeviceConnection(this) {
    m_material = std::make_shared<CMaterial>(CMaterial(COrenNayarBRDF(diffuseReflection, diffuseRougness), CMicrofacetBRDF(specularReflection, alphaX, alphaY, etaI, etaT)));
  }

  CHostSceneobject::CHostSceneobject(CShape* shape, CMedium* medium):
    m_shape(shape),
    m_mesh(nullptr),
    m_material(nullptr),
    m_medium(medium),
    m_flag(ESceneobjectFlag::VOLUME),
    m_deviceGasBuffer(NULL),
    m_hostDeviceConnection(this) {
  }

  CHostSceneobject::CHostSceneobject(CNVDBMedium* medium) :
    m_shape(nullptr),
    m_mesh(nullptr),
    m_material(nullptr),
    m_medium(medium),
    m_flag(ESceneobjectFlag::VOLUME),
    m_deviceGasBuffer(NULL),
    m_hostDeviceConnection(this) {
  }

  CHostSceneobject::CHostSceneobject(CHostSceneobject&& sceneobject) :
    m_shape(std::move(sceneobject.m_shape)),
    m_mesh(std::move(sceneobject.m_mesh)),
    m_material(std::move(sceneobject.m_material)),
    m_medium(std::move(sceneobject.m_medium)),
    m_flag(std::move(sceneobject.m_flag)),
    m_deviceGasBuffer(std::exchange(sceneobject.m_deviceGasBuffer, NULL)),
    m_hostDeviceConnection(this) {
  }

  CHostSceneobject::CHostSceneobject(CMesh* mesh, const glm::vec3& diffuseReflection, float diffuseRougness, const glm::vec3& specularReflection, float alphaX, float alphaY, float etaI, float etaT) :
    m_shape(nullptr),
    m_mesh(mesh),
    m_material(nullptr),
    m_medium(nullptr),
    m_flag(ESceneobjectFlag::GEOMETRY),
    m_deviceGasBuffer(NULL),
    m_hostDeviceConnection(this) {
    CUDA_LOG_ERROR_STATE();
    m_material = std::make_shared<CMaterial>(CMaterial(COrenNayarBRDF(diffuseReflection, diffuseRougness), CMicrofacetBRDF(specularReflection, alphaX, alphaY, etaI, etaT)));
    CUDA_LOG_ERROR_STATE();
  }

  CSceneobjectConnection::CSceneobjectConnection(CHostSceneobject* hostSceneobject):
    m_hostSceneobject(hostSceneobject) {
  }

  CSceneobjectConnection::CSceneobjectConnection(const CSceneobjectConnection&& connection) :
    m_hostSceneobject(std::move(connection.m_hostSceneobject)) {
  }

  void CSceneobjectConnection::allocateDeviceMemory() {
    CUDA_LOG_ERROR_STATE();
    if (m_hostSceneobject->m_shape) {
      switch (m_hostSceneobject->m_shape->shape()) {
      case EShape::CIRCLE:
        cudaMalloc(&m_deviceShape, sizeof(CCircle));
        break;
      case EShape::SPHERE:
        cudaMalloc(&m_deviceShape, sizeof(Sphere));
        break;
      case EShape::RECTANGLE:
        cudaMalloc(&m_deviceShape, sizeof(CRectangle));
        break;
      case EShape::CUBOID:
        cudaMalloc(&m_deviceShape, sizeof(CCuboid));
        break;
      }
    }
    if (m_hostSceneobject->m_mesh) {
      CUDA_LOG_ERROR_STATE();
      cudaMalloc(&m_deviceMesh, sizeof(CMesh));
      m_hostSceneobject->m_mesh->allocateDeviceMemory();
      CUDA_LOG_ERROR_STATE();
    }
    if (m_hostSceneobject->m_material) {
      cudaMalloc(&m_deviceMaterial, sizeof(CMaterial));
    }
    if (m_hostSceneobject->m_medium) {
      switch (m_hostSceneobject->m_medium->type()) {
      case EMediumType::HOMOGENEOUS_MEDIUM:
        cudaMalloc(&m_deviceMedium, sizeof(CHomogeneousMedium));
        break;
      case EMediumType::HETEROGENOUS_MEDIUM:
        cudaMalloc(&m_deviceMedium, sizeof(CHeterogenousMedium));
        std::static_pointer_cast<CHeterogenousMedium>(m_hostSceneobject->m_medium)->allocateDeviceMemory();
        break;
      case EMediumType::NVDB_MEDIUM:
        cudaMalloc(&m_deviceMedium, sizeof(CNVDBMedium));
        std::static_pointer_cast<CNVDBMedium>(m_hostSceneobject->m_medium)->allocateDeviceMemory();
        break;
      }
    }
    
    CUDA_LOG_ERROR_STATE();
  }
  void CSceneobjectConnection::copyToDevice() {
    if (m_deviceShape) {
      switch (m_hostSceneobject->m_shape->shape()) {
      case EShape::CIRCLE:
        cudaMemcpy(m_deviceShape, m_hostSceneobject->m_shape.get(), sizeof(CCircle), cudaMemcpyHostToDevice);
        break;
      case EShape::SPHERE:
        cudaMemcpy(m_deviceShape, m_hostSceneobject->m_shape.get(), sizeof(Sphere), cudaMemcpyHostToDevice);
        break;
      case EShape::RECTANGLE:
        cudaMemcpy(m_deviceShape, m_hostSceneobject->m_shape.get(), sizeof(CRectangle), cudaMemcpyHostToDevice);
        break;
      case EShape::CUBOID:
        cudaMemcpy(m_deviceShape, m_hostSceneobject->m_shape.get(), sizeof(CCuboid), cudaMemcpyHostToDevice);
        break;
      }
    }
    if (m_deviceMesh) {
      cudaMemcpy(m_deviceMesh, &m_hostSceneobject->m_mesh->copyToDevice(), sizeof(CMesh), cudaMemcpyHostToDevice);
    }
    if (m_deviceMaterial) {
      cudaMemcpy(m_deviceMaterial, m_hostSceneobject->m_material.get(), sizeof(CMaterial), cudaMemcpyHostToDevice);
    }
    if (m_deviceMedium) {
      switch (m_hostSceneobject->m_medium->type()) {
      case EMediumType::HOMOGENEOUS_MEDIUM:
        cudaMemcpy(m_deviceMedium, m_hostSceneobject->m_medium.get(), sizeof(CHomogeneousMedium), cudaMemcpyHostToDevice);
        break;
      case EMediumType::HETEROGENOUS_MEDIUM: {
        std::shared_ptr<CHeterogenousMedium> hetMedium = std::static_pointer_cast<CHeterogenousMedium>(m_hostSceneobject->m_medium);
        cudaMemcpy(m_deviceMedium, &hetMedium->copyToDevice(), sizeof(CHeterogenousMedium), cudaMemcpyHostToDevice);
        break;
      }
      case EMediumType::NVDB_MEDIUM: {
        std::shared_ptr<CNVDBMedium> nvdbMedium = std::static_pointer_cast<CNVDBMedium>(m_hostSceneobject->m_medium);
        cudaMemcpy(m_deviceMedium, &nvdbMedium->copyToDevice(), sizeof(CNVDBMedium), cudaMemcpyHostToDevice);
        break;
      }
      }
    }
    if (m_deviceSceneobject) {

      CDeviceSceneobject deviceSceneobject;
      deviceSceneobject.m_shape = m_deviceShape;
      deviceSceneobject.m_mesh = m_deviceMesh;
      deviceSceneobject.m_material = m_deviceMaterial;
      deviceSceneobject.m_medium = m_deviceMedium;
      deviceSceneobject.m_flag = m_hostSceneobject->m_flag;
      cudaMemcpy(m_deviceSceneobject, &deviceSceneobject, sizeof(CDeviceSceneobject), cudaMemcpyHostToDevice);
    }
  }

  void CSceneobjectConnection::freeDeviceMemory() {
    cudaFree(m_deviceShape);
    if (m_deviceMesh) {
      m_hostSceneobject->m_mesh->freeDeviceMemory();
      cudaFree(m_deviceMesh);
    }
    cudaFree(m_deviceMaterial);
    if (m_deviceMedium) {
      switch (m_hostSceneobject->m_medium->type()) {
        case EMediumType::HETEROGENOUS_MEDIUM: {
          std::shared_ptr<CHeterogenousMedium> hetMedium = std::static_pointer_cast<CHeterogenousMedium>(m_hostSceneobject->m_medium);
          hetMedium->freeDeviceMemory();
          break;
        }
        case EMediumType::NVDB_MEDIUM: {
          std::shared_ptr<CNVDBMedium> nvdbMedium = std::static_pointer_cast<CNVDBMedium>(m_hostSceneobject->m_medium);
          nvdbMedium->freeDeviceMemory();
          break;
        }
      }
      cudaFree(m_deviceMedium);
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
    CUDA_LOG_ERROR_STATE();
    cudaFree((void*)m_deviceGasBuffer);
    CUDA_LOG_ERROR_STATE();
  }

  void CHostSceneobject::buildOptixAccel() {
    SBuildInputWrapper buildInputWrapper;
    if (m_medium.get() && m_medium->type() == NVDB_MEDIUM) {
      buildInputWrapper = ((CNVDBMedium*)m_medium.get())->getOptixBuildInput();
    }
    else if (m_mesh) {
      buildInputWrapper = m_mesh->getOptixBuildInput();
    }
    else if (m_shape) {
      buildInputWrapper = m_shape->getOptixBuildInput();
    }
    else {
      fprintf(stderr, "[ERROR] Could not create build input.\n");
    }

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gasBufferSizes;
    const OptixDeviceContext& context = CRTBackend::instance()->context();
    OPTIX_ASSERT(optixAccelComputeMemoryUsage(context, &accelOptions, &buildInputWrapper.buildInput, 1, &gasBufferSizes));
    CUDA_LOG_ERROR_STATE();

    CUdeviceptr d_tempBufferGas;
    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void**>(&d_tempBufferGas), gasBufferSizes.tempSizeInBytes));
    CUdeviceptr d_outputBufferGas;
    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void**>(&d_outputBufferGas), gasBufferSizes.outputSizeInBytes));
    CUdeviceptr d_compactedSize;
    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void**>(&d_compactedSize), sizeof(size_t)));

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = d_compactedSize;

    printf("CUDA error state: %s\n", cudaGetErrorString(cudaGetLastError()));
    CUDA_LOG_ERROR_STATE();

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

  OptixInstance CHostSceneobject::getOptixInstance(uint32_t instanceId, uint32_t sbtOffset) const {
    OptixInstance instance;

    instance.flags = OPTIX_INSTANCE_FLAG_NONE;
    instance.instanceId = instanceId;
    instance.sbtOffset = sbtOffset;
    instance.visibilityMask = 0xff; // TODO: Check what has to be set here
    instance.traversableHandle = m_traversableHandle;
    float identity[] = { 1.f, 0.f, 0.f, 0.f,
                         0.f, 1.f, 0.f, 0.f,
                         0.f, 0.f, 1.f, 0.f };
    memcpy(instance.transform, identity, sizeof(float) * 12);

    return instance;
  }

  OptixProgramGroup CHostSceneobject::getOptixProgramGroup() const {
    if (m_medium.get()) {
      return m_medium->getOptixProgramGroup();
    }
    else if (m_shape.get()) {
      return m_shape->getOptixProgramGroup();
    }
    else if (m_mesh.get()) {
      return m_mesh->getOptixProgramGroup();
    }
    fprintf(stderr, "[ERROR] CHostSceneobject::getOptixProgramGroup no valid program group found.\n");
    return OptixProgramGroup();
  }

  SRecord<const CDeviceSceneobject*> CHostSceneobject::getSBTHitRecord() const {
    SRecord<const CDeviceSceneobject*> hitRecord;
    OPTIX_ASSERT(optixSbtRecordPackHeader(getOptixProgramGroup(), &hitRecord));
    CUDA_LOG_ERROR_STATE();
    if (!m_hostDeviceConnection.deviceSceneobject()) {
      fprintf(stderr, "[ERROR] CHostSceneobject::getSBTHitRecord: deviceSceneobject is null.\n");
    }
    hitRecord.data = m_hostDeviceConnection.deviceSceneobject();
    return hitRecord;
  }


}