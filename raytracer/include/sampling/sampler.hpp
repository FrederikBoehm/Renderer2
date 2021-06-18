#ifndef SAMPLER_HPP
#define SAMPLER_HPP

#include "curand_kernel.h"
#include "utility/qualifiers.hpp"
#include <glm/glm.hpp>

namespace rt {
  class CSampler {
  public:
    D_CALLABLE void init();

    D_CALLABLE float uniformSample01();
    D_CALLABLE glm::vec3 uniformSampleHemisphere();
    D_CALLABLE float uniformHemispherePdf() const;
  private:
    curandState_t m_curandState;
  };
}


#endif // !SAMPLE_HEMISPHERE_HPP
