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

  float CSampler::uniformSample01() {
    // TODO: Make sampler work on host
#ifdef __CUDA_ARCH__
    return curand_uniform(&m_curandState);
#else 
    return 0.0f;
#endif
  }

  glm::vec3 CSampler::uniformSampleHemisphere() {
    float rand1 = curand_uniform(&m_curandState);
    float rand2 = curand_uniform(&m_curandState);

    float r = glm::sqrt(glm::max(0.0f, 1.0f - rand1 * rand1));
    float phi = 2.0 * M_PI * rand2;
    return glm::vec3(r * glm::cos(phi), rand1, r * glm::sin(phi));
  }

  float CSampler::uniformHemispherePdf() const {
    return 1.0f / (2 * M_PI);
  }

}
