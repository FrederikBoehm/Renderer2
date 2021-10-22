#ifndef INTEGRATOR_OBJECTS_HPP
#define INTEGRATOR_OBJECTS_HPP
#include <glm/glm.hpp>
#include "utility/qualifiers.hpp"
namespace rt {
  class CCoordinateFrame {
  public:
    DH_CALLABLE const glm::vec3& N() const {
      return m_N;
    }

    DH_CALLABLE const glm::vec3& T() const {
      return m_T;
    }

    DH_CALLABLE const glm::vec3& B() const {
      return m_B;
    }

    DH_CALLABLE const glm::mat4& tangentToWorld() const {
      return m_tangentToWorld;
    }

    DH_CALLABLE const glm::mat4& worldToTangent() const {
      return m_worldToTangent;
    }

    DH_CALLABLE static CCoordinateFrame fromNormal(const glm::vec3& N) {
      // Construct tangent space
      CCoordinateFrame frame;
      frame.m_N = N;
      glm::vec3 notN = glm::normalize(glm::vec3(N.x + 1.0f, N.x + 2.0f, N.x + 3.0f));
      frame.m_T = glm::normalize(glm::cross(notN, N));
      frame.m_B = glm::normalize(glm::cross(N, frame.m_T));
      frame.m_tangentToWorld = glm::mat4(glm::vec4(frame.m_T, 0.0f), glm::vec4(frame.m_B, 0.0f), glm::vec4(N, 0.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
      frame.m_worldToTangent = glm::inverse(frame.m_tangentToWorld);
      return frame;
    }

  private:
    glm::vec3 m_N;
    glm::vec3 m_T;
    glm::vec3 m_B;

    glm::mat4 m_tangentToWorld;
    glm::mat4 m_worldToTangent;

    DH_CALLABLE CCoordinateFrame() {}

  };
}
#endif // !OBJECTS_HPP
