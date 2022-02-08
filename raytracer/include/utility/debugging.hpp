#ifndef DEBUGGING_HPP
#define DEBUGGING_HPP
#include "cuda_runtime.h"
#include <stdio.h>
#include "qualifiers.hpp"
#include <optix/optix_types.h>
#include <optix/optix_host.h>

inline void gpuAssert(cudaError_t code, const char* file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA_ASSERT: %s in file %s on line %i\n", cudaGetErrorString(code), file, line);
  }
}
#define CUDA_ASSERT(ans) {gpuAssert((ans), __FILE__, __LINE__);}

inline void logCudaErrorState(const char* file, int line) {
  printf("CUDA error state: %s in file %s on line %i\n", cudaGetErrorString(cudaGetLastError()), file, line);
}
#define CUDA_LOG_ERROR_STATE() {logCudaErrorState(__FILE__, __LINE__);}

inline void optixAssert(OptixResult res, const char* file, int line) {
  if (res != OPTIX_SUCCESS) {
    fprintf(stderr, "OPTIX_ASSERT: %s in file %s on line %i\n", optixGetErrorString(res), file, line);
  }
}
#define OPTIX_ASSERT(ans) {optixAssert((ans), __FILE__, __LINE__);}

namespace rt {
  class CSampler;
  H_CALLABLE void storeRandomState(CSampler* samplers, size_t numStates, const char* storePath);
  H_CALLABLE void loadRandomState(CSampler* samplers, const char* loadPath);
}

#endif
