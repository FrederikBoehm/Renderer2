#ifndef PIXEL_SAMPLER_HPP
#define PIXEL_SAMPLER_HPP
#include <stdint.h>
#include <glm/glm.hpp>

#include "utility/qualifiers.hpp"
#include "intersect/ray.hpp"

namespace rt {
  class CCamera;
  class CSampler;

  class CPixelSampler {
  public:
    D_CALLABLE CPixelSampler(CCamera* camera, uint16_t pixelX, uint16_t pixelY, CSampler* sampler);

    /*
      Samples a pixel randomly. Returns a ray in world space.
      Origin (0, 0) is bottom left.
    */
    D_CALLABLE Ray samplePixel() const;

  private:
    CCamera* m_camera;
    uint16_t m_pixelX;
    uint16_t m_pixelY;
    CSampler* m_sampler;
  };
}
#endif // !PIXEL_SAMPLER_HPP
