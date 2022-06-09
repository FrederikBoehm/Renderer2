#ifndef SAMPLER_HPP
#define SAMPLER_HPP
#define _USE_MATH_DEFINES
#include <cmath>

#include "curand_kernel.h"
#include "utility/qualifiers.hpp"
#include <glm/glm.hpp>

namespace rt {
  class CSampler {
    __global__ friend void copyStates(CSampler* samplers, curandState_t* states, size_t numStates);
    __global__ friend void copyStates(curandState_t* states, CSampler* samplers, size_t numStates);

  public:
    D_CALLABLE void init();
    D_CALLABLE void init(uint64_t seed, uint64_t sequence);

    DH_CALLABLE float uniformSample01();
    D_CALLABLE glm::vec3 uniformSampleHemisphere();
    D_CALLABLE static float uniformHemispherePdf();
    D_CALLABLE glm::vec3 concentricSampleDisk();
    D_CALLABLE glm::vec3 cosineSampleHemisphere();
    D_CALLABLE static float cosineHemispherePdf(float cosTheta);
    D_CALLABLE glm::vec3 uniformSampleSphere();
    D_CALLABLE static float uniformSpherePdf();
  private:
    curandState_t m_curandState;
  };

  inline float CSampler::uniformSample01() { // produces random numbers in the interval [0, 1)
#ifdef __CUDA_ARCH__
    return glm::clamp(curand_uniform(&m_curandState) / (1.f - FLT_EPSILON - FLT_MIN) - FLT_MIN, 0.f, 1.f - FLT_EPSILON); // Because curand_unfiform produces random numbers in (0.f, 1.f] we have to scale and shift to get numbers in [0.f, 1.f)
#else 
    return 0.0f;
#endif
  }

  inline glm::vec3 CSampler::uniformSampleHemisphere() {
    float rand1 = uniformSample01();
    float rand2 = uniformSample01();

    float r = glm::sqrt(glm::max(0.0f, 1.0f - rand1 * rand1));
    float phi = 2.0 * M_PI * rand2;
    return glm::vec3(r * glm::cos(phi), r * glm::sin(phi), rand1);
  }

  inline float CSampler::uniformHemispherePdf() {
    return 1.0f / (2 * M_PI);
  }

  inline glm::vec3 CSampler::concentricSampleDisk() {
    float rand1 = uniformSample01();
    float rand2 = uniformSample01();

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

  inline glm::vec3 CSampler::cosineSampleHemisphere() {
    glm::vec3 d = concentricSampleDisk();
    float z = glm::sqrt(glm::max(0.f, 1.f - d.x * d.x - d.y * d.y));
    return glm::vec3(d.x, d.y, z);
  }

  inline float CSampler::cosineHemispherePdf(float cosTheta) {
    return cosTheta * M_1_PI;
  }

  inline glm::vec3 CSampler::uniformSampleSphere() {
    float u1 = uniformSample01();
    float u2 = uniformSample01();

    float z = 1.f - 2.f * u1;
    float r = glm::sqrt(glm::max(0.f, 1.f - z * z));
    float phi = 2 * M_PI * u2;
    return glm::vec3(r * glm::cos(phi), r * glm::sin(phi), z);
  }

  inline float CSampler::uniformSpherePdf() {
    return 1.f / (4.f * M_PI);
  }

}


#endif // !SAMPLE_HEMISPHERE_HPP
