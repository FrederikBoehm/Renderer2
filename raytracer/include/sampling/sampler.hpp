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
    D_CALLABLE float uniformHemispherePdf() const;
    D_CALLABLE glm::vec3 concentricSampleDisk();
    D_CALLABLE glm::vec3 cosineSampleHemisphere();
    D_CALLABLE float cosineHemispherePdf(float cosTheta);
    D_CALLABLE glm::vec3 uniformSampleSphere();
    D_CALLABLE float uniformSpherePdf() const;
  private:
    curandState_t m_curandState;
  };

}


#endif // !SAMPLE_HEMISPHERE_HPP
