#ifndef RAY_HPP
#define RAY_HPP

#include "utility/qualifiers.hpp"

#include <glm/glm.hpp>
namespace rt {
  class CMedium;

  class CRay {
  public:
    inline static constexpr float OFFSET = 1e-4f;
    inline static constexpr float DEFAULT_TMAX = 1e+12;
    DH_CALLABLE CRay(const glm::vec3& origin, const glm::vec3& direction, float t_max = DEFAULT_TMAX, const CMedium* medium = nullptr);
    glm::vec3 m_origin;
    glm::vec3 m_direction;
    mutable float m_t_max;
    const CMedium* m_medium;
    DH_CALLABLE static glm::vec3 offsetRayOrigin(const glm::vec3& p, const glm::vec3& offsetDir);
    DH_CALLABLE CRay offsetRayOrigin(const glm::vec3& offsetDir);
    DH_CALLABLE CRay transform(const glm::mat4& worldToModel) const;
    DH_CALLABLE CRay robustTransform(const glm::mat4& worldToModel, const glm::vec3& offsetDir) const;
    DH_CALLABLE static CRay spawnRay(const glm::vec3& start, const glm::vec3& end, const CMedium* originMedium = nullptr);
  };

  
}

#endif