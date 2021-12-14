#ifndef PIXEL_SAMPLER_HPP
#define PIXEL_SAMPLER_HPP
#include <stdint.h>
#include <glm/glm.hpp>

#include "utility/qualifiers.hpp"
#include "intersect/ray.hpp"
#include "sampling/sampler.hpp"
#include "camera/camera.hpp"

namespace rt {

  class CPixelSampler {
  public:
    D_CALLABLE CPixelSampler(CCamera* camera, uint16_t pixelX, uint16_t pixelY, CSampler* sampler) :
      m_camera(camera),
      m_pixelX(pixelX),
      m_pixelY(pixelY),
      m_sampler(sampler) {

    }

    /*
      Samples a pixel randomly. Returns a ray in world space.
      Origin (0, 0) is bottom left.
    */
    D_CALLABLE CRay samplePixel() const;

  private:
    CCamera* m_camera;
    uint16_t m_pixelX;
    uint16_t m_pixelY;
    CSampler* m_sampler;
  };

  inline CRay CPixelSampler::samplePixel() const {
    float randomHorizontal = m_sampler->uniformSample01();
    float horizontal = (m_pixelX - m_camera->m_sensorWidth / 2 + randomHorizontal) * m_camera->m_pixelSize;
    float randomVertical = m_sampler->uniformSample01();
    float vertical = (m_pixelY - m_camera->m_sensorHeight / 2 + randomHorizontal) * m_camera->m_pixelSize;
    float depth = -m_camera->m_nearPlaneDistance;

    glm::vec4 rayDir = glm::vec4(glm::normalize(glm::vec3(horizontal, vertical, depth)), 0.0f);
    glm::vec4 rayOrigin = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    CRay viewSpaceRay(rayOrigin, rayDir);
    return viewSpaceRay.transform(m_camera->m_viewToWorld);
  }
}
#endif // !PIXEL_SAMPLER_HPP
