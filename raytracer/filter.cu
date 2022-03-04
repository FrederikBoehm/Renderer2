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
    m_clipRays(config.filteringConfig.clipRays) {
    SOpenvdbBackendConfig openvdbConfig;
    auto[modelSpaceBBs, worldSpaceBBs] = config.scene->getObjectBBs(rt::ESceneobjectMask::FILTER);
    openvdbConfig.modelSpaceBoundingBoxes = modelSpaceBBs;
    openvdbConfig.worldSpaceBoundingBoxes = worldSpaceBBs;
    openvdbConfig.voxelSize = config.filteringConfig.voxelSize;
    openvdbConfig.debug = config.filteringConfig.debug;
    if (openvdbConfig.modelSpaceBoundingBoxes.size() > 0) {
      m_backend = filter::COpenvdbBackend::instance();
      m_backend->init(openvdbConfig);
      
      allocateDeviceMemory();
      initOptix(config);
      copyToDevice();
      initDeviceData();
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
    const SFilterLaunchParams& launchParams = m_backend->launchParams();
    dim3 grid(launchParams.numVoxels.x, launchParams.numVoxels.y, launchParams.numVoxels.z); // TODO: Find way to fully utilize blocks with variable numVoxels
    init << <grid, 1 >> > (m_deviceSampler);
    CUDA_ASSERT(cudaDeviceSynchronize());
  }

  void CFilter::runFiltering() const {
    if (!m_backend) {
      return;
    }
    const glm::ivec3& numVoxels = m_backend->launchParams().numVoxels;
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
      std::vector<SFilteredData> filteredData(voxelCount);
      CUDA_ASSERT(cudaMemcpy(filteredData.data(), m_deviceFilterData, sizeof(SFilteredData) * voxelCount, cudaMemcpyDeviceToHost));
      m_backend->setValues(filteredData);
      nanovdb::GridHandle<nanovdb::HostBuffer> gridHandle = m_backend->getNanoGridHandle();
      m_backend->writeToFile(gridHandle, "./filtering", "filtered_mesh.nvdb");
    }
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
    const glm::ivec3& numVoxels = m_backend->launchParams().numVoxels;
    size_t totalVoxels = numVoxels.x * numVoxels.y * numVoxels.z;
    CUDA_ASSERT(cudaMalloc(&m_deviceSampler, sizeof(rt::CSampler) * totalVoxels));
    CUDA_ASSERT(cudaMalloc(&m_deviceFilterData, sizeof(SFilteredData) * totalVoxels))
  }

  void CFilter::copyToDevice() {
    SFilterLaunchParams launchParams = m_backend->launchParams();
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
    CUDA_ASSERT(cudaMemcpy(m_deviceLaunchParams, &launchParams, sizeof(SFilterLaunchParams), cudaMemcpyHostToDevice));
  }

  void CFilter::freeDeviceMemory() {
    CUDA_ASSERT(cudaFree(m_deviceLaunchParams));
    CUDA_ASSERT(cudaFree(m_deviceSampler));
    CUDA_ASSERT(cudaFree(m_deviceFilterData));
  }
}