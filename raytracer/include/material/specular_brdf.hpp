#ifndef SPECULAR_BRDF_H
#define SPECULAR_BRDF_H

#include <glm/glm.hpp>
#include "utility/qualifiers.hpp"


class CSpecularBRDF {
  D_CALLABLE glm::vec3 f(const glm::vec3& wo, const glm::vec3& wi) {
    return glm::vec3(0.0f);
  }

};

#endif // !SPECULAR_BRDF_H

