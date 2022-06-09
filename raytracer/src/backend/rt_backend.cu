#include "backend/rt_backend.hpp"
#include "utility/debugging.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>



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
    CUDA_ASSERT(cudaFree(reinterpret_cast<void*>(m_sbt.raygenRecord)));
    CUDA_ASSERT(cudaFree(reinterpret_cast<void*>(m_sbt.missRecordBase)));
    CUDA_ASSERT(cudaFree(reinterpret_cast<void*>(m_sbt.hitgroupRecordBase)));

    OPTIX_ASSERT(optixPipelineDestroy(m_pipeline)); 

    OPTIX_ASSERT(optixProgramGroupDestroy(m_programGroups.m_raygen));
    OPTIX_ASSERT(optixProgramGroupDestroy(m_programGroups.m_miss));
    OPTIX_ASSERT(optixProgramGroupDestroy(m_programGroups.m_hitSurface));
    OPTIX_ASSERT(optixProgramGroupDestroy(m_programGroups.m_hitVolume));
    OPTIX_ASSERT(optixProgramGroupDestroy(m_programGroups.m_hitMesh));

    OPTIX_ASSERT(optixModuleDestroy(m_module));
    OPTIX_ASSERT(optixDeviceContextDestroy(m_context));
  }

  void CRTBackend::reset() {
    CUDA_ASSERT(cudaFree(reinterpret_cast<void*>(m_sbt.raygenRecord)));
    CUDA_ASSERT(cudaFree(reinterpret_cast<void*>(m_sbt.missRecordBase)));
    CUDA_ASSERT(cudaFree(reinterpret_cast<void*>(m_sbt.hitgroupRecordBase)));
    m_sbt.raygenRecord = NULL;
    m_sbt.missRecordBase = NULL;
    m_sbt.hitgroupRecordBase = NULL;

    if (m_pipeline) {
      OPTIX_ASSERT(optixPipelineDestroy(m_pipeline));
      m_pipeline = nullptr;
    }

    if (m_programGroups.m_raygen) {
      OPTIX_ASSERT(optixProgramGroupDestroy(m_programGroups.m_raygen));
      m_programGroups.m_raygen = nullptr;
    }
    if (m_programGroups.m_miss) {
      OPTIX_ASSERT(optixProgramGroupDestroy(m_programGroups.m_miss));
      m_programGroups.m_miss = nullptr;
    }
    if (m_programGroups.m_hitSurface) {
      OPTIX_ASSERT(optixProgramGroupDestroy(m_programGroups.m_hitSurface));
      m_programGroups.m_hitSurface = nullptr;
    }
    if (m_programGroups.m_hitVolume) {
      OPTIX_ASSERT(optixProgramGroupDestroy(m_programGroups.m_hitVolume));
      m_programGroups.m_hitVolume = nullptr;
    }
    if (m_programGroups.m_hitMesh) {
      OPTIX_ASSERT(optixProgramGroupDestroy(m_programGroups.m_hitMesh));
      m_programGroups.m_hitMesh = nullptr;
    }

    if (m_module) {
      OPTIX_ASSERT(optixModuleDestroy(m_module));
      m_module = nullptr;
    }
  }

  CRTBackend* CRTBackend::instance() {
    if (s_instance == nullptr) {
      s_instance = new CRTBackend();
    }
    return s_instance;
  }

  OptixPipelineCompileOptions CRTBackend::getCompileOptions(const char* launchParams) {
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING | OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues = 9;
    pipelineCompileOptions.numAttributeValues = 7;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = launchParams;
    return pipelineCompileOptions;
  }

  void CRTBackend::createModule(const std::string& ptxFile, const char* launchParams) {
    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    OptixPipelineCompileOptions pipelineCompileOptions = getCompileOptions(launchParams);

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

  void CRTBackend::createProgramGroups(const SProgramGroupFunctionNames& functionNames) {
    OptixProgramGroupOptions programGroupOptions = {};

    char log[2048];
    size_t sizeofLog = sizeof(log);

    // Raygen
    {
      OptixProgramGroupDesc raygenProgGroupDesc = {};
      raygenProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
      raygenProgGroupDesc.raygen.module = m_module;
      raygenProgGroupDesc.raygen.entryFunctionName = functionNames.raygen;

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
      missProgGroupDesc.miss.entryFunctionName = functionNames.miss;
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
      hitProgGroupDesc.hitgroup.entryFunctionNameCH = functionNames.closesthit;
      hitProgGroupDesc.hitgroup.moduleAH = nullptr;
      hitProgGroupDesc.hitgroup.entryFunctionNameAH = nullptr;
      hitProgGroupDesc.hitgroup.moduleIS = m_module;
      hitProgGroupDesc.hitgroup.entryFunctionNameIS = functionNames.intersectSurface;
      sizeofLog = sizeof(log);
      OPTIX_ASSERT(optixProgramGroupCreate(
        m_context,
        &hitProgGroupDesc,
        1,   // num program groups
        &programGroupOptions,
        log,
        &sizeofLog,
        &m_programGroups.m_hitSurface
      ));
    }

    // Hitgroup volume
    {
      OptixProgramGroupDesc hitProgGroupDesc = {};
      hitProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
      hitProgGroupDesc.hitgroup.moduleCH = m_module;
      hitProgGroupDesc.hitgroup.entryFunctionNameCH = functionNames.closesthit;
      hitProgGroupDesc.hitgroup.moduleAH = nullptr;
      hitProgGroupDesc.hitgroup.entryFunctionNameAH = nullptr;
      hitProgGroupDesc.hitgroup.moduleIS = m_module;
      hitProgGroupDesc.hitgroup.entryFunctionNameIS = functionNames.intersectionVolume;
      sizeofLog = sizeof(log);
      OPTIX_ASSERT(optixProgramGroupCreate(
        m_context,
        &hitProgGroupDesc,
        1,   // num program groups
        &programGroupOptions,
        log,
        &sizeofLog,
        &m_programGroups.m_hitVolume
      ));
    }

    // Hitgroup mesh
    {
      OptixProgramGroupDesc hitProgGroupDesc = {};
      hitProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
      hitProgGroupDesc.hitgroup.moduleCH = m_module;
      hitProgGroupDesc.hitgroup.entryFunctionNameCH = functionNames.closesthit;
      //hitProgGroupDesc.hitgroup.moduleAH = nullptr;
      //hitProgGroupDesc.hitgroup.entryFunctionNameAH = nullptr;
      hitProgGroupDesc.hitgroup.moduleAH = m_module;
      hitProgGroupDesc.hitgroup.entryFunctionNameAH = functionNames.anyhitMesh;
      hitProgGroupDesc.hitgroup.moduleIS = nullptr;
      hitProgGroupDesc.hitgroup.entryFunctionNameIS = nullptr;
      sizeofLog = sizeof(log);
      OPTIX_ASSERT(optixProgramGroupCreate(
        m_context,
        &hitProgGroupDesc,
        1,
        &programGroupOptions,
        log,
        &sizeofLog,
        &m_programGroups.m_hitMesh
      ));
    }

  }

  void CRTBackend::createPipeline(const char* launchParams) {
    OptixPipelineCompileOptions pipelineCompileOptions = getCompileOptions(launchParams);

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
    OPTIX_ASSERT(optixUtilAccumulateStackSizes(m_programGroups.m_hitSurface, &stackSizes));
    OPTIX_ASSERT(optixUtilAccumulateStackSizes(m_programGroups.m_hitVolume, &stackSizes));
    OPTIX_ASSERT(optixUtilAccumulateStackSizes(m_programGroups.m_hitMesh, &stackSizes));

    uint32_t maxTraceDepth = 4; 
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
      const size_t hitgroupRecordSize = sizeof(SRecord<const CDeviceSceneobject*>);
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