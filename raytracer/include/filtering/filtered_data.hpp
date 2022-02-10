#ifndef FILTERED_DATA_HPP
#define FILTERED_DATA_HPP
#include <glm/glm.hpp>

namespace filter {
  struct SFilteredData {
    float density;
    float specularRoughness;
    glm::u16vec3 diffuseColor;
    glm::u16vec3 specularColor;
    glm::vec3 normal;
  };
}

#endif // !FILTERED_DATA_HPP
