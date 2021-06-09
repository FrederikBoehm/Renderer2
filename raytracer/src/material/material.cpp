#include "material/material.hpp"

namespace rt {
  CMaterial::CMaterial() {
  }
  CMaterial::CMaterial(const glm::vec3& albedo):
    m_albedo(albedo) {

  }
}