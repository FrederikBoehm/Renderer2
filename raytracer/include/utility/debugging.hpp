#ifndef DEBUGGING_HPP
#define DEBUGGING_HPP
#include "cuda_runtime.h"
#include <stdio.h>
#include "qualifiers.hpp"

namespace rt {
  #define GPU_ASSERT(ans) {gpuAssert((ans), __FILE__, __LINE__);}
  inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
      fprintf(stderr, "GPU_ASSERT: %s in file %s on line %i\n", cudaGetErrorString(code), file, line);
    }
  }

  class CSampler;
  H_CALLABLE void storeRandomState(CSampler* samplers, size_t numStates, const char* storePath);
  H_CALLABLE void loadRandomState(CSampler* samplers, const char* loadPath);
}

#endif
