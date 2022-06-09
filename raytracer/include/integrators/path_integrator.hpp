#ifndef PATH_INTEGRATOR_HPP
#define PATH_INTEGRATOR_HPP

#include <glm/glm.hpp>
#include "utility/qualifiers.hpp"
#include "backend/types.hpp"


namespace rt {

  class CDeviceScene;
  class CSampler;
  class CPixelSampler;

  class CPathIntegrator { 
  public:
    D_CALLABLE CPathIntegrator(CDeviceScene* scene, CPixelSampler* pixelSampler, CSampler* sampler, uint16_t numSamples, bool useBrickGrid, EDebugMode debugMode);
    D_CALLABLE glm::vec3 Li() const;

  private:
    CDeviceScene* m_scene;
    CPixelSampler* m_pixelSampler;
    CSampler* m_sampler;
    uint16_t m_numSamples;
    bool m_useBrickGrid;
    
    EDebugMode m_debugMode;
  };
}
#endif // !PATH_INTEGRATOR_HPP

