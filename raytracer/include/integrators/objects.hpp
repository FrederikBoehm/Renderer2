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
      frame.m_T = glm::abs(N.x) > glm::abs(N.y) ?
        glm::vec3(-N.z, 0.f, N.x) / glm::sqrt(N.x * N.x + N.z * N.z) :
        glm::vec3(0.f, N.z, -N.y) / glm::sqrt(N.y * N.y + N.z * N.z);
      frame.m_B = glm::normalize(glm::cross(N, frame.m_T));
      frame.m_tangentToWorld = glm::mat4(glm::vec4(frame.m_T, 0.0f), glm::vec4(frame.m_B, 0.0f), glm::vec4(N, 0.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
      frame.m_worldToTangent = glm::inverse(frame.m_tangentToWorld);
      return frame;
    }

    DH_CALLABLE static CCoordinateFrame fromTBN(const glm::vec3& T, const glm::vec3& B, const glm::vec3& N) {
      CCoordinateFrame frame;
      frame.m_N = N;
      frame.m_T = T;
      frame.m_B = B;
      frame.m_tangentToWorld = glm::mat4(glm::vec4(frame.m_T, 0.0f), glm::vec4(frame.m_B, 0.0f), glm::vec4(frame.m_N, 0.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
      frame.m_worldToTangent = glm::inverse(frame.m_tangentToWorld);
      return frame;
    }

    DH_CALLABLE static glm::vec3 align(const glm::vec3& axis, const glm::vec3& v) {
      const float s = copysign(1.f, axis.z);
      const glm::vec3 w = glm::vec3(v.x, v.y, v.z * s);
      const glm::vec3 h = glm::vec3(axis.x, axis.y, axis.z + s);
      const float k = glm::dot(w, h) / (1.f + fabsf(axis.z));
      return k * h - w;
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
