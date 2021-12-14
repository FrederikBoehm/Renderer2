#include "utility/debugging.hpp"
#include "sampling/sampler.hpp"
#include <string>
#include <vector>
#include <fstream>

namespace rt {
  __global__ void copyStates(CSampler* samplers, curandState_t* states, size_t numStates) {
    for (size_t i = 0; i < numStates; ++i) {
      states[i] = samplers[i].m_curandState;
    }
  }

  __global__ void copyStates(curandState_t* states, CSampler* samplers, size_t numStates) {
    for (size_t i = 0; i < numStates; ++i) {
      samplers[i].m_curandState = states[i];
    }
  }

  void storeRandomState(CSampler* samplers, size_t numStates, const char* storePath) {
    curandState_t* d_states;
    cudaMalloc(&d_states, sizeof(curandState_t) * numStates);

    copyStates << <1, 1 >> > (samplers, d_states, numStates);
    CUDA_ASSERT(cudaDeviceSynchronize());
    std::vector<curandState_t> localStates(numStates);
    cudaMemcpy(localStates.data(), d_states, sizeof(curandState_t) * numStates, cudaMemcpyDeviceToHost);

    cudaFree(d_states);

    std::fstream s(storePath, std::ios_base::binary | std::ios_base::out);
    s.write((const char*)localStates.data(), sizeof(curandState_t) * localStates.size());
  }

  void loadRandomState(CSampler* samplers, const char* loadPath) {
    std::ifstream file(loadPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<curandState_t> buffer(size / sizeof(curandState_t));
    file.read((char*)buffer.data(), size);

    curandState_t* d_states;
    cudaMalloc(&d_states, size);

    cudaMemcpy(d_states, buffer.data(), size, cudaMemcpyHostToDevice);

    copyStates << <1, 1 >> > (d_states, samplers, size / sizeof(curandState_t));
    CUDA_ASSERT(cudaDeviceSynchronize());

    cudaFree(d_states);
  }
}