#define _USE_MATH_DEFINES
#include <cmath>
#include "sampling/sampler.hpp"

namespace rt {
  void CSampler::init() {
    curand_init(0, 0, 0, &m_curandState);
  }

  void CSampler::init(uint64_t seed, uint64_t sequence) {
    curand_init(seed, sequence, 0, &m_curandState);
  }


}
