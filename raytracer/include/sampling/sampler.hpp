#ifndef SAMPLER_HPP
#define SAMPLER_HPP
#define _USE_MATH_DEFINES
#include <cmath>

#include "curand_kernel.h"
#include "utility/qualifiers.hpp"
#include <glm/glm.hpp>

namespace rt {
  class CSampler {
  public:
    D_CALLABLE void init();
    D_CALLABLE void init(uint64_t seed, uint64_t sequence);

    D_CALLABLE float uniformSample01();
    D_CALLABLE glm::vec3 uniformSampleHemisphere();
    D_CALLABLE float uniformHemispherePdf() const;
  private:
    curandState_t m_curandState;
  };

  //inline void CSampler::init() {
  //  curand_init(0, 0, 0, &m_curandState);
  //}

  //inline void CSampler::init(uint64_t seed, uint64_t sequence) {
  //  curand_init(seed, sequence, 0, &m_curandState);
  //}

  //inline float CSampler::uniformSample01() {
  //  return curand_uniform(&m_curandState);
  //}

  //inline glm::vec3 CSampler::uniformSampleHemisphere() {
  //  float rand1 = curand_uniform(&m_curandState);
  //  float rand2 = curand_uniform(&m_curandState);

  //  float r = glm::sqrt(glm::max(0.0f, 1.0f - rand1 * rand1));
  //  float phi = 2.0 * M_PI * rand2;
  //  return glm::vec3(r * glm::cos(phi), r * glm::sin(phi), rand1);
  //}

  //inline float CSampler::uniformHemispherePdf() const {
  //  return 1.0f / (2 * M_PI);
  //}
}


#endif // !SAMPLE_HEMISPHERE_HPP
