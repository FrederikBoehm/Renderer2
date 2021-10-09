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
    return glm::vec3(r * glm::cos(phi), r * glm::sin(phi), rand1);
  }

  float CSampler::uniformHemispherePdf() const {
    return 1.0f / (2 * M_PI);
  }

  glm::vec3 CSampler::concentricSampleDisk() {
    float rand1 = curand_uniform(&m_curandState);
    float rand2 = curand_uniform(&m_curandState);

    glm::vec2 uOffset = 2.0f * glm::vec2(rand1, rand2) - glm::vec2(1.0f);

    if (uOffset.x == 0 && uOffset.y == 0) {
      return glm::vec3(0.0f, 0.0f, 0.0f);
    }

    float theta;
    float r;
    if (glm::abs(uOffset.x) > glm::abs(uOffset.y)) {
      r = uOffset.x;
      theta = M_PI_4 * (uOffset.y / uOffset.x);
    }
    else {
      r = uOffset.y;
      theta = M_PI_2 - M_PI_4 * (uOffset.x / uOffset.y);
    }

    return r * glm::vec3(glm::cos(theta), glm::sin(theta), 0.0f);
  }

  glm::vec3 CSampler::cosineSampleHemisphere() {
    glm::vec3 d = concentricSampleDisk();
    float z = glm::sqrt(glm::max(0.f, 1.f - d.x * d.x - d.y * d.y));
    return glm::vec3(d.x, d.y, z);
  }

  float CSampler::cosineHemispherePdf(float cosTheta) {
    return cosTheta * M_1_PI;
  }

}
