#ifndef MATERIAL_HXX
#define MATERIAL_HXX
#include <glm/glm.hpp>

#include "utility/qualifiers.hpp"
namespace rt {
  class CMaterial {
  public:
    CMaterial();
    CMaterial(const glm::vec3& albedo);

    DH_CALLABLE const glm::vec3& albedo() const;

  private:
    glm::vec3 m_albedo;
  };


  inline const glm::vec3& CMaterial::albedo() const {
    return m_albedo;
  }

}
#endif // !MATERIAL_HXX
