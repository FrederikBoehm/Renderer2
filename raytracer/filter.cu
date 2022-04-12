#include "filter.hpp"
#include "backend/config_loader.hpp"
#include "filtering/openvdb_backend.hpp"
#include "backend/rt_backend.hpp"
#include "utility/debugging.hpp"
#include <optix/optix_stubs.h>
#include "backend/types.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "filtering/launch_params.hpp"
#include "filtering/volume_description_manager.hpp"
#include "utility/functions.hpp"

namespace filter {
  CFilter::CFilter(const SConfig& config):
    m_backend(nullptr),
    m_deviceLaunchParams(nullptr),
    m_deviceScene(config.scene->deviceScene()),
    m_deviceFilterData(nullptr),
    m_samplesPerVoxel(config.filteringConfig.samplesPerVoxel),
    m_debug(config.filteringConfig.debug),
    m_debugSamples(config.filteringConfig.debugSamples),
    m_sigma_t(config.filteringConfig.sigmaT),
    m_estimationIterations(config.filteringConfig.estimationIterations),
    m_alpha(config.filteringConfig.alpha),
    m_clipRays(config.filteringConfig.clipRays),
    m_voxelSize(config.filteringConfig.voxelSize),
    m_lods(config.filteringConfig.lods){
    SOpenvdbBackendConfig openvdbConfig;
    auto[modelSpaceBBs, worldSpaceBBs, worldToModel, filename, orientation, scaling] = config.scene->getObjectBBs(rt::ESceneobjectMask::FILTER);
    openvdbConfig.modelSpaceBoundingBoxes = modelSpaceBBs;
    openvdbConfig.worldSpaceBoundingBoxes = worldSpaceBBs;
    openvdbConfig.worldToModel = worldToModel;
    openvdbConfig.voxelSize = config.filteringConfig.voxelSize;
    m_outDir = "./filtering/" + filename;
    m_filename = filename;
    m_orientation = orientation;
    m_scaling = scaling;
    if (openvdbConfig.modelSpaceBoundingBoxes.size() > 0) {
      m_backend = filter::COpenvdbBackend::instance();
      m_backend->init(openvdbConfig);
      
      allocateDeviceMemory();
      initOptix(config);
      initDeviceData();
      CVolumeDescriptionManager::instance()->loadDescriptions("../../raytracer/src/filtering/volume_description.json");
    }
    else {
      printf("[WARNING]: No bounding boxes provided --> proceed without filtering.\n");
    }

  }

  CFilter::~CFilter() {
  }

  __global__ void init(rt::CSampler* sampler) {
    size_t samplerId = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    sampler[samplerId].init(samplerId, 0);
  }

  void CFilter::initDeviceData() const {
    const glm::ivec3& numVoxels = m_backend->numVoxelsMajorant();
    dim3 grid(numVoxels.x, numVoxels.y, numVoxels.z); // TODO: Find way to fully utilize blocks with variable numVoxels
    init << <grid, 1 >> > (m_deviceSampler);
    CUDA_ASSERT(cudaDeviceSynchronize());
  }

  void CFilter::runFiltering() const {
    if (!m_backend) {
      return;
    }

    CVolumeDescriptionManager* volumeDescriptionManager = CVolumeDescriptionManager::instance();

    uint8_t lod;
    float voxelSize;
    for (lod = 0, voxelSize = m_voxelSize; lod < m_lods; ++lod, voxelSize *= 2.f) {
      SFilterLaunchParams launchParams = m_backend->setupGrid(glm::vec3(voxelSize));
      copyToDevice(launchParams);

      const glm::ivec3& numVoxels = launchParams.numVoxels;
      OPTIX_ASSERT(optixLaunch(
        rt::CRTBackend::instance()->pipeline(),
        0,             // stream
        reinterpret_cast<CUdeviceptr>(m_deviceLaunchParams),
        sizeof(filter::SFilterLaunchParams),
        &rt::CRTBackend::instance()->sbt(),
        numVoxels.x,  // launch width
        numVoxels.y, // launch height
        numVoxels.z       // launch depth
      ));
      CUDA_ASSERT(cudaDeviceSynchronize());

      if (!m_debug) {
        size_t voxelCount = numVoxels.x * numVoxels.y * numVoxels.z;
        std::vector<SFilteredDataCompact> filteredData(voxelCount);
        CUDA_ASSERT(cudaMemcpy(filteredData.data(), m_deviceFilterData, sizeof(SFilteredDataCompact) * voxelCount, cudaMemcpyDeviceToHost));
        m_backend->setValues(filteredData, numVoxels);
        nanovdb::GridHandle<nanovdb::HostBuffer> gridHandle = m_backend->getNanoGridHandle();
        std::string filename = "filtered_mesh_" + std::to_string(voxelSize) + ".nvdb";
        m_backend->writeToFile(gridHandle, m_outDir.c_str(), filename.c_str());


        glm::mat4 transform = rt::getRotation(m_orientation) * glm::mat4(launchParams.worldToModel);
        glm::vec3 tempMin = transform * glm::vec4(launchParams.worldBB.m_min, 1.f);
        glm::vec3 tempMax = transform * glm::vec4(launchParams.worldBB.m_max, 1.f);
        glm::vec3 modelMin = glm::min(tempMin, tempMax);
        glm::vec3 modelMax = glm::max(tempMin, tempMax);
        glm::vec3 dimensions = modelMax - modelMin;
        volumeDescriptionManager->addDescription(m_filename, voxelSize, modelMin, modelMax, numVoxels);
      }
    }

    volumeDescriptionManager->storeDescriptions("../../raytracer/src/filtering/volume_description.json");
  }

  void CFilter::initOptix(const SConfig& config) {
    rt::CRTBackend* rtBackend = rt::CRTBackend::instance();
    rtBackend->init();
#ifdef DEBUG
    std::string modulePath = "cuda_to_ptx.dir/Debug/shaders.optix.ptx";
#endif
#ifdef RELEASE
    std::string modulePath = "cuda_to_ptx.dir/Release/shaders.optix.ptx";
#endif
    rtBackend->createModule(modulePath, "paramsFiltering");
    rtBackend->createProgramGroups({ "__raygen__filtering", "__miss__ms", "__closesthit__ch", "__intersection__surface", "__intersection__volume", "__anyhit__mesh" });
    rtBackend->createPipeline("paramsFiltering");
    const std::vector <rt::SRecord<const rt::CDeviceSceneobject*>> sbtHitRecords =config.scene->getSBTHitRecords();
    rtBackend->createSBT(sbtHitRecords);
    config.scene->buildOptixAccel();
  }

  void CFilter::allocateDeviceMemory() {
    CUDA_ASSERT(cudaMalloc(&m_deviceLaunchParams, sizeof(SFilterLaunchParams)));
    const glm::ivec3& numVoxels = m_backend->numVoxelsMajorant();
    size_t totalVoxels = numVoxels.x * numVoxels.y * numVoxels.z;
    CUDA_ASSERT(cudaMalloc(&m_deviceSampler, sizeof(rt::CSampler) * totalVoxels));
    CUDA_ASSERT(cudaMalloc(&m_deviceFilterData, sizeof(SFilteredDataCompact) * totalVoxels))
  }

  void CFilter::copyToDevice(SFilterLaunchParams& launchParams) const {
    launchParams.samplers = m_deviceSampler;
    launchParams.scene = m_deviceScene;
    launchParams.samplesPerVoxel = m_samplesPerVoxel;
    launchParams.filteredData = m_deviceFilterData;
    launchParams.debug = m_debug;
    launchParams.debugSamples = m_debugSamples;
    launchParams.sigma_t = m_sigma_t;
    launchParams.estimationIterations = m_estimationIterations;
    launchParams.alpha = m_alpha;
    launchParams.clipRays = m_clipRays;
    launchParams.scaling = m_scaling.x;
    CUDA_ASSERT(cudaMemcpy(m_deviceLaunchParams, &launchParams, sizeof(SFilterLaunchParams), cudaMemcpyHostToDevice));
  }

  void CFilter::freeDeviceMemory() {
    CUDA_ASSERT(cudaFree(m_deviceLaunchParams));
    CUDA_ASSERT(cudaFree(m_deviceSampler));
    CUDA_ASSERT(cudaFree(m_deviceFilterData));
  }
}