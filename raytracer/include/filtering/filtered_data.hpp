#ifndef FILTERED_DATA_HPP
#define FILTERED_DATA_HPP
#include <glm/glm.hpp>

namespace filter {
  struct SFilteredDataCompact;

  struct SFilteredData { // For filtering, voxel interpolation and rendering
    float density;
    glm::mat3 S;
    glm::vec3 diffuseColor;
    glm::vec3 specularColor;
    
    DH_CALLABLE SFilteredData() :
      density(0.f),
      S(0.f),
      diffuseColor(0.f),
      specularColor(0.f) {

    }

    DH_CALLABLE SFilteredData(const SFilteredDataCompact& data);
  };

  struct SFilteredDataCompact { // For storing
    float density;
    uint16_t sigma_x; // [0, 1] -> [0, U16_MAX]
    uint16_t sigma_y; // [0, 1] -> [0, U16_MAX]
    uint16_t sigma_z; // [0, 1] -> [0, U16_MAX]
    uint16_t r_xy; // [-1, 1] -> [0, U16_MAX]
    uint16_t r_xz; // [-1, 1] -> [0, U16_MAX]
    uint16_t r_yz; // [-1, 1] -> [0, U16_MAX]
    glm::u16vec3 diffuseColor;
    glm::u16vec3 specularColor;

    DH_CALLABLE SFilteredDataCompact() :
      density(0.f),
      sigma_x(0),
      sigma_y(0),
      sigma_z(0),
      r_xy(0),
      r_xz(0),
      r_yz(0),
      diffuseColor(0),
      specularColor(0) {

    }

    DH_CALLABLE SFilteredDataCompact(const SFilteredData& data);
  };

  inline SFilteredData::SFilteredData(const SFilteredDataCompact& data) {
    density = data.density;
    uint16_t MAX_U16 = -1;
    float fMAX_U16 = MAX_U16;
    float sigma_x = data.sigma_x / fMAX_U16;
    float sigma_y = data.sigma_y / fMAX_U16;
    float sigma_z = data.sigma_z / fMAX_U16;
    float r_xy = 2.f * (data.r_xy / fMAX_U16) - 1.f;
    float r_xz = 2.f * (data.r_xz / fMAX_U16) - 1.f;
    float r_yz = 2.f * (data.r_yz / fMAX_U16) - 1.f;

    float S_xx = sigma_x * sigma_x;
    float S_yy = sigma_y * sigma_y;
    float S_zz = sigma_z * sigma_z;
    float S_xy = r_xy * sigma_x * sigma_y;
    float S_xz = r_xz * sigma_x * sigma_z;
    float S_yz = r_yz * sigma_y * sigma_z;
    glm::mat3 tempS(S_xx, S_xy, S_xz,
                    S_xy, S_yy, S_yz,
                    S_xz, S_yz, S_zz);
    S = tempS;
    diffuseColor = glm::vec3(data.diffuseColor) / fMAX_U16;
    specularColor = glm::vec3(data.specularColor) / fMAX_U16;
  }

  inline SFilteredDataCompact::SFilteredDataCompact(const SFilteredData& data) {
    density = data.density;
    uint16_t MAX_U16 = -1;
    float fMAX_U16 = MAX_U16;
    sigma_x = glm::clamp(fMAX_U16 * glm::sqrt(data.S[0][0]), 0.f, fMAX_U16);
    sigma_y = glm::clamp(fMAX_U16 * glm::sqrt(data.S[1][1]), 0.f, fMAX_U16);
    sigma_z = glm::clamp(fMAX_U16 * glm::sqrt(data.S[2][2]), 0.f, fMAX_U16);
    r_xy = glm::clamp(fMAX_U16 * 0.5f * (data.S[0][1] / glm::sqrt(data.S[0][0] * data.S[1][1]) + 1.f), 0.f, fMAX_U16);
    r_xz = glm::clamp(fMAX_U16 * 0.5f * (data.S[0][2] / glm::sqrt(data.S[0][0] * data.S[2][2]) + 1.f), 0.f, fMAX_U16);
    r_yz = glm::clamp(fMAX_U16 * 0.5f * (data.S[1][2] / glm::sqrt(data.S[1][1] * data.S[2][2]) + 1.f), 0.f, fMAX_U16);
    diffuseColor = glm::clamp(fMAX_U16 * data.diffuseColor, 0.f, fMAX_U16);
    specularColor = glm::clamp(fMAX_U16 * data.specularColor, 0.f, fMAX_U16);
  }


  // Operators for interpolation
  DH_CALLABLE inline SFilteredData operator*(float v, const SFilteredData& filteredData) {
    SFilteredData out;
    out.density = v * filteredData.density;
    float S_xx = v * filteredData.S[0][0];
    float S_yy = v * filteredData.S[1][1];
    float S_zz = v * filteredData.S[2][2];
    float S_xy = v * filteredData.S[0][1];
    float S_xz = v * filteredData.S[0][2];
    float S_yz = v * filteredData.S[1][2];
    glm::mat3 S(S_xx, S_xy, S_xz,
                S_xy, S_yy, S_yz,
                S_xz, S_yz, S_zz);
    out.S = S;
    out.diffuseColor = v * filteredData.diffuseColor;
    out.specularColor = v * filteredData.specularColor;
    return out;
  }

  DH_CALLABLE inline SFilteredData operator+(const SFilteredData& v1, const SFilteredData& v2) {
    SFilteredData out;
    out.density = v1.density + v2.density;
    float S_xx = v1.S[0][0] + v2.S[0][0];
    float S_yy = v1.S[1][1] + v2.S[1][1];
    float S_zz = v1.S[2][2] + v2.S[2][2];
    float S_xy = v1.S[0][1] + v2.S[0][1];
    float S_xz = v1.S[0][2] + v2.S[0][2];
    float S_yz = v1.S[1][2] + v2.S[1][2];
    glm::mat3 S(S_xx, S_xy, S_xz,
                S_xy, S_yy, S_yz,
                S_xz, S_yz, S_zz);
    out.S = S;
    out.diffuseColor = v1.diffuseColor + v2.diffuseColor;
    out.specularColor = v1.specularColor + v2.specularColor;
    return out;
  }
}

#endif // !FILTERED_DATA_HPP
