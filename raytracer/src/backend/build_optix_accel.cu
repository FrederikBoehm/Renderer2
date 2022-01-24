#include "backend/build_optix_accel.hpp"
#include <optix/optix_stubs.h>
#include "utility/debugging.hpp"
#include "backend/rt_backend.hpp"

namespace rt {
  void buildOptixAccel(const OptixBuildInput& buildInput, OptixTraversableHandle* traversableHandle, CUdeviceptr* deviceGasBuffer) {
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gasBufferSizes;
    const OptixDeviceContext& context = CRTBackend::instance()->context();
    OPTIX_ASSERT(optixAccelComputeMemoryUsage(context, &accelOptions, &buildInput, 1, &gasBufferSizes));

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
      &buildInput,
      1,
      d_tempBufferGas,
      gasBufferSizes.tempSizeInBytes,
      d_outputBufferGas,
      gasBufferSizes.outputSizeInBytes,
      traversableHandle,
      &emitProperty,
      1));

    CUDA_ASSERT(cudaStreamSynchronize(0));

    size_t compactedSize;
    CUDA_ASSERT(cudaMemcpy(&compactedSize, reinterpret_cast<void*>(emitProperty.result), sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaFree(reinterpret_cast<void*>(d_compactedSize)));
    if (compactedSize < gasBufferSizes.outputSizeInBytes)
    {
      CUDA_ASSERT(cudaMalloc(reinterpret_cast<void**>(deviceGasBuffer), compactedSize));
      OPTIX_ASSERT(optixAccelCompact(context, 0, *traversableHandle, *deviceGasBuffer, compactedSize, traversableHandle));
      CUDA_ASSERT(cudaFree(reinterpret_cast<void*>(d_outputBufferGas)));
    }
    else
    {
      *deviceGasBuffer = d_outputBufferGas;
    }
    CUDA_ASSERT(cudaFree(reinterpret_cast<void*>(d_tempBufferGas)));
  }
}