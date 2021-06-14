#ifndef SPHERE_HXX
#define SPHERE_HXX

#include "utility/qualifiers.hpp"

#include "shape.hpp"
#include "intersect/hit_information.hpp"
#include "intersect/ray.hpp"
namespace rt {
  class Sphere : public CShape {
  public:
    DH_CALLABLE Sphere();
    DH_CALLABLE Sphere(float radius);
    DH_CALLABLE Sphere(const glm::vec3& worldPos, float radius);

    DH_CALLABLE SHitInformation intersect(const Ray& ray) const;


  private:
    float m_radius;
  };

}
#endif // !SPHERE_HXX
