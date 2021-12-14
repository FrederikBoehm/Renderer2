#include "backend/rt_backend.hpp"
#include "utility/debugging.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <optix/optix_stubs.h>
#include <optix/optix_function_table_definition.h>
#include <optix/optix_stack_size.h>



namespace rt {
  CRTBackend* CRTBackend::s_instance = nullptr;

  H_CALLABLE void optixContextLog(unsigned int level, const char* tag, const char* message, void* /*cbdata */) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
  }

  void CRTBackend::init() {
    CUDA_ASSERT(cudaFree(0));
    OPTIX_ASSERT(optixInit());
    OptixDeviceContextOptions options = {};
#define DEBUG_OPTIX
#ifdef DEBUG_OPTIX
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
    options.logCallbackFunction = &optixContextLog;
    options.logCallbackLevel = 4;
    OPTIX_ASSERT(optixDeviceContextCreate(0, &options, &m_context));
  }

  void CRTBackend::release() {
    OPTIX_ASSERT(optixDeviceContextDestroy(m_context));
  }

  CRTBackend* CRTBackend::instance() {
    if (s_instance == nullptr) {
      s_instance = new CRTBackend();
    }
    return s_instance;
  }

  OptixPipelineCompileOptions CRTBackend::getCompileOptions() {
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING | OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues = 9;
    pipelineCompileOptions.numAttributeValues = 7;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    return pipelineCompileOptions;
  }

  void CRTBackend::createModule(const std::string& ptxFile) {
    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    OptixPipelineCompileOptions pipelineCompileOptions = getCompileOptions();

    char log[2048];
    size_t sizeofLog = sizeof(log);

    std::fstream ptxFileStream(ptxFile, std::ios_base::in);
    std::stringstream buffer;
    buffer << ptxFileStream.rdbuf();
    std::string input = buffer.str();

    OPTIX_ASSERT(optixModuleCreateFromPTX(
      m_context,
      &moduleCompileOptions,
      &pipelineCompileOptions,
      input.c_str(),
      input.size(),
      log,
      &sizeofLog,
      &m_module
    ));

    std::cout << log << std::endl;
  }

  void CRTBackend::createProgramGroups() {
    OptixProgramGroupOptions programGroupOptions = {};

    char log[2048];
    size_t sizeofLog = sizeof(log);

    // Raygen
    {
      OptixProgramGroupDesc raygenProgGroupDesc = {};
      raygenProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
      raygenProgGroupDesc.raygen.module = m_module;
      raygenProgGroupDesc.raygen.entryFunctionName = "__raygen__rg";

      OPTIX_ASSERT(optixProgramGroupCreate(
        m_context,
        &raygenProgGroupDesc,
        1,                             // num program groups
        &programGroupOptions,
        log,
        &sizeofLog,
        &m_programGroups.m_raygen
      ));
    }

    // Miss
    {
      OptixProgramGroupDesc missProgGroupDesc = {};
      missProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
      missProgGroupDesc.miss.module = m_module;
      missProgGroupDesc.miss.entryFunctionName = "__miss__ms";
      sizeofLog = sizeof(log);
      OPTIX_ASSERT(optixProgramGroupCreate(
        m_context,
        &missProgGroupDesc,
        1,                             // num program groups
        &programGroupOptions,
        log,
        &sizeofLog,
        &m_programGroups.m_miss
      ));
    }

    // Hitgroup circle
    {
      OptixProgramGroupDesc hitProgGroupDesc = {};
      hitProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
      hitProgGroupDesc.hitgroup.moduleCH = m_module;
      hitProgGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
      hitProgGroupDesc.hitgroup.moduleAH = nullptr;
      hitProgGroupDesc.hitgroup.entryFunctionNameAH = nullptr;
      hitProgGroupDesc.hitgroup.moduleIS = m_module;
      hitProgGroupDesc.hitgroup.entryFunctionNameIS = "__intersection__surface";
      sizeofLog = sizeof(log);
      OPTIX_ASSERT(optixProgramGroupCreate(
        m_context,
        &hitProgGroupDesc,
        1,   // num program groups
        &programGroupOptions,
        log,
        &sizeofLog,
        &m_programGroups.m_intersectSurface
      ));
    }

    // Hitgroup volume
    {
      OptixProgramGroupDesc hitProgGroupDesc = {};
      hitProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
      hitProgGroupDesc.hitgroup.moduleCH = m_module;
      hitProgGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
      hitProgGroupDesc.hitgroup.moduleAH = nullptr;
      hitProgGroupDesc.hitgroup.entryFunctionNameAH = nullptr;
      hitProgGroupDesc.hitgroup.moduleIS = m_module;
      hitProgGroupDesc.hitgroup.entryFunctionNameIS = "__intersection__volume";
      sizeofLog = sizeof(log);
      OPTIX_ASSERT(optixProgramGroupCreate(
        m_context,
        &hitProgGroupDesc,
        1,   // num program groups
        &programGroupOptions,
        log,
        &sizeofLog,
        &m_programGroups.m_intersectVolume
      ));
    }

  }

  void CRTBackend::createPipeline() {
    OptixPipelineCompileOptions pipelineCompileOptions = getCompileOptions();

    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 4;
    pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    char log[2048];
    size_t sizeofLog = sizeof(log);
    OPTIX_ASSERT(optixPipelineCreate(
      m_context,
      &pipelineCompileOptions,
      &pipelineLinkOptions,
      &m_programGroups.m_raygen,                      // ptr to first program group
      sizeof(SProgramGroups) / sizeof(OptixProgramGroup), // number of program groups
      log,
      &sizeofLog,
      &m_pipeline
    ));

    OptixStackSizes stackSizes = {};
    OPTIX_ASSERT(optixUtilAccumulateStackSizes(m_programGroups.m_raygen, &stackSizes));
    OPTIX_ASSERT(optixUtilAccumulateStackSizes(m_programGroups.m_miss, &stackSizes));
    OPTIX_ASSERT(optixUtilAccumulateStackSizes(m_programGroups.m_intersectSurface, &stackSizes));
    OPTIX_ASSERT(optixUtilAccumulateStackSizes(m_programGroups.m_intersectVolume, &stackSizes));

    uint32_t maxTraceDepth = 4; // TODO: Is that the path tracing depth?
    uint32_t maxCcDepth = 0;
    uint32_t maxDcDepth = 4;
    uint32_t directCallableStackSizeFromTraversal;
    uint32_t directCallableStackSizeFromState;
    uint32_t continuationStackSize;
    OPTIX_ASSERT(optixUtilComputeStackSizes(
      &stackSizes,
      maxTraceDepth,
      maxCcDepth,
      maxDcDepth,
      &directCallableStackSizeFromTraversal,
      &directCallableStackSizeFromState,
      &continuationStackSize
    ));

    const uint32_t maxTraversalDepth = 2;
    OPTIX_ASSERT(optixPipelineSetStackSize(
      m_pipeline,
      directCallableStackSizeFromTraversal,
      directCallableStackSizeFromState,
      continuationStackSize,
      maxTraversalDepth
    ));
  }

  void CRTBackend::createSBT(const std::vector<SRecord<const CDeviceSceneobject*>>& hitgroupRecords) {
    {
      const size_t raygenRecordSize = sizeof(SRecord<SEmptyData>);
      CUDA_ASSERT(cudaMalloc(reinterpret_cast<void**>(&m_sbt.raygenRecord), raygenRecordSize));

      SRecord<SEmptyData> rg_sbt;
      OPTIX_ASSERT(optixSbtRecordPackHeader(m_programGroups.m_raygen, &rg_sbt));
      CUDA_ASSERT(cudaMemcpy(
        reinterpret_cast<void*>(m_sbt.raygenRecord),
        &rg_sbt,
        raygenRecordSize,
        cudaMemcpyHostToDevice
      ));
    }

    {

      const size_t missRecordSize = sizeof(SRecord<SEmptyData>);
      CUDA_ASSERT(cudaMalloc(
        reinterpret_cast<void**>(&m_sbt.missRecordBase),
        missRecordSize * RAY_TYPE_COUNT
      ));

      SRecord<SEmptyData> ms_sbt[RAY_TYPE_COUNT];
      OPTIX_ASSERT(optixSbtRecordPackHeader(m_programGroups.m_miss, &ms_sbt[0]));

      CUDA_ASSERT(cudaMemcpy(
        reinterpret_cast<void*>(m_sbt.missRecordBase),
        ms_sbt,
        missRecordSize * RAY_TYPE_COUNT,
        cudaMemcpyHostToDevice
      ));
      m_sbt.missRecordStrideInBytes = static_cast<uint32_t>(missRecordSize);
      m_sbt.missRecordCount = RAY_TYPE_COUNT;
    }

    {
      const size_t hitgroupRecordSize = sizeof(SRecord<const CDeviceSceneobject*>); // TODO: Pass SInteraction
      const size_t hitgroupRecordBytes = hitgroupRecordSize * hitgroupRecords.size();
      CUDA_ASSERT(cudaMalloc(
        reinterpret_cast<void**>(&m_sbt.hitgroupRecordBase),
        hitgroupRecordBytes
      ));

      CUDA_ASSERT(cudaMemcpy(
        reinterpret_cast<void*>(m_sbt.hitgroupRecordBase),
        hitgroupRecords.data(),
        hitgroupRecordBytes,
        cudaMemcpyHostToDevice
      ));
      m_sbt.hitgroupRecordStrideInBytes = hitgroupRecordSize;
      m_sbt.hitgroupRecordCount = hitgroupRecords.size();
    }

  }

}