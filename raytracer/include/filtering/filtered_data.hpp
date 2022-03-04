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

  // Operators for interpolation
  DH_CALLABLE inline SFilteredData operator*(float v, const SFilteredData& filteredData) {
    SFilteredData out;
    out.density = v * filteredData.density;
    out.specularRoughness = v * filteredData.specularRoughness;
    out.diffuseColor = glm::round(v * glm::vec3(filteredData.diffuseColor));
    out.specularColor = glm::round(v * glm::vec3(filteredData.specularColor));
    out.normal = v * filteredData.normal;
    return out;
  }

  DH_CALLABLE inline SFilteredData operator+(const SFilteredData& v1, const SFilteredData& v2) {
    SFilteredData out;
    out.density = v1.density + v2.density;
    out.specularRoughness = v1.specularRoughness + v2.specularRoughness;
    out.diffuseColor = v1.diffuseColor + v2.diffuseColor;
    out.specularColor = v1.specularColor + v2.specularColor;
    out.normal = v1.normal + v2.normal;
    return out;
  }
}

#endif // !FILTERED_DATA_HPP
