#include "camera/pixel_sampler.hpp"
#include "sampling/sampler.hpp"
#include "camera/camera.hpp"

namespace rt {
  CPixelSampler::CPixelSampler(CCamera* camera, uint16_t pixelX, uint16_t pixelY, CSampler* sampler) :
    m_camera(camera),
    m_pixelX(pixelX),
    m_pixelY(pixelY),
    m_sampler(sampler) {

  }

  Ray CPixelSampler::samplePixel() const {
    float randomHorizontal = m_sampler->uniformSample01();
    float horizontal = (m_pixelX - m_camera->m_sensorWidth / 2 + randomHorizontal) * m_camera->m_pixelSize;
    float randomVertical = m_sampler->uniformSample01();
    float vertical = (m_pixelY - m_camera->m_sensorHeight / 2 + randomHorizontal) * m_camera->m_pixelSize;
    float depth = -m_camera->m_nearPlaneDistance;

    glm::vec4 rayDir = glm::vec4(glm::normalize(glm::vec3(horizontal, vertical, depth)), 0.0f);
    glm::vec4 rayOrigin = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    Ray viewSpaceRay(rayOrigin, rayDir);
    return viewSpaceRay.transform(m_camera->m_viewToWorld);
  }
}