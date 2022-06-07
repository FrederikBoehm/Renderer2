#ifndef DEBUGGING_HPP
#define DEBUGGING_HPP
#include "cuda_runtime.h"
#include <stdio.h>
#include "qualifiers.hpp"
#include <optix_types.h>
#include <optix_host.h>
#include <glm/glm.hpp>

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

  DH_CALLABLE inline glm::vec3 mapLODs(float lodSize) {
    if (lodSize == 0.1f) {
      return glm::vec3(0.f, 0.f, 1.f);
    }
    else if (lodSize == 0.2f) {
      return glm::vec3(0.f, 0.67f, 1.f);
    }
    else if (lodSize == 0.4f) {
      return glm::vec3(0.f, 1.f, 0.67f);
    }
    else if (lodSize == 0.8f) {
      return glm::vec3(0.f, 1.f, 0.f);
    }
    else if (lodSize == 1.6f) {
      return glm::vec3(0.67f, 1.f, 0.f);
    }
    else if (lodSize == 3.2f) {
      return glm::vec3(1.f, 0.67f, 0.f);
    }
    else {
      return glm::vec3(1.f, 0.f, 0.f);
    }
  }
}

#endif
