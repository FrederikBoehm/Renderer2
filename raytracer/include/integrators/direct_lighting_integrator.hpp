#ifndef DIRECT_LIGHTING_INTEGRATOR_HPP
#define DIRECT_LIGHTING_INTEGRATOR_HPP

#include <glm/glm.hpp>
#include "utility/qualifiers.hpp"


namespace rt {
  enum EIntegrationStrategy {
    UNIFORM_SAMPLE_HEMISPHERE,
    IMPORTANCE_SAMPLE_LIGHTSOURCES
  };

  class CDeviceScene;
  class CSampler;
  class CPixelSampler;

  class CDirectLightingIntegrator { // TODO: Make Integrator host callable
  public:
    D_CALLABLE CDirectLightingIntegrator(CDeviceScene* scene, CPixelSampler* pixelSampler, CSampler* sampler, uint16_t numSamples);
    D_CALLABLE glm::vec3 Li(EIntegrationStrategy strategy);

  private:
    CDeviceScene* m_scene;
    CPixelSampler* m_pixelSampler;
    CSampler* m_sampler;
    uint16_t m_numSamples;
  };
}
#endif // !DIRECT_LIGHTING_INTEGRATOR_HPP

