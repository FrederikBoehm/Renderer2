#ifndef FILTERED_DATA_HPP
#define FILTERED_DATA_HPP
#include <glm/glm.hpp>
#include <cuda_fp16.h>

namespace filter {
  struct SFilteredDataCompact;

  struct SFilteredData { // For filtering, voxel interpolation and rendering
    float density;
    glm::mat3 S;
    glm::vec3 diffuseColor;
    glm::vec3 specularColor;
    glm::vec3 normal; // Might not have length 1 during interpolation
    float ior;
    
    DH_CALLABLE SFilteredData() :
      density(0.f),
      S(0.f),
      diffuseColor(0.f),
      specularColor(0.f),
      normal(0.f),
      ior(0.f) {

    }

    DH_CALLABLE SFilteredData(const SFilteredDataCompact& data);

    DH_CALLABLE glm::vec3 n() const;
  };

  struct SFilteredDataCompact { // For storing
    float density;
    uint16_t sigma_x; // [0, 1] -> [0, U16_MAX]
    uint16_t sigma_y; // [0, 1] -> [0, U16_MAX]
    uint16_t sigma_z; // [0, 1] -> [0, U16_MAX]
    uint16_t r_xy; // [-1, 1] -> [0, U16_MAX]
    uint16_t r_xz; // [-1, 1] -> [0, U16_MAX]
    uint16_t r_yz; // [-1, 1] -> [0, U16_MAX]
    glm::u8vec3 diffuseColor;
    glm::u8vec3 specularColor;
    glm::u16vec3 normal;
    __half ior;

    DH_CALLABLE SFilteredDataCompact() :
      density(0.f),
      sigma_x(0),
      sigma_y(0),
      sigma_z(0),
      r_xy(0),
      r_xz(0),
      r_yz(0),
      diffuseColor(0),
      specularColor(0),
      normal(0),
      ior(0.f) {
    }

    DH_CALLABLE SFilteredDataCompact(const SFilteredData& data);
  };

  inline SFilteredData::SFilteredData(const SFilteredDataCompact& data) {
    density = data.density;
    uint16_t MAX_U16 = -1;
    float invfMAX_U16 = 1.f / (float)MAX_U16;
    uint8_t MAX_U8 = -1;
    float invfMAX_U8 = 1.f / (float)MAX_U8;
    float sigma_x = data.sigma_x * invfMAX_U16;
    float sigma_y = data.sigma_y * invfMAX_U16;
    float sigma_z = data.sigma_z * invfMAX_U16;
    float r_xy = 2.f * (data.r_xy * invfMAX_U16) - 1.f;
    float r_xz = 2.f * (data.r_xz * invfMAX_U16) - 1.f;
    float r_yz = 2.f * (data.r_yz * invfMAX_U16) - 1.f;

    float S_xx = sigma_x * sigma_x;
    float S_yy = sigma_y * sigma_y;
    float S_zz = sigma_z * sigma_z;
    float S_xy = r_xy * sigma_x * sigma_y;
    float S_xz = r_xz * sigma_x * sigma_z;
    float S_yz = r_yz * sigma_y * sigma_z;
    S = glm::mat3(S_xx, S_xy, S_xz,
                  S_xy, S_yy, S_yz,
                  S_xz, S_yz, S_zz);
    diffuseColor = glm::vec3(data.diffuseColor) * invfMAX_U8;
    specularColor = glm::vec3(data.specularColor) * invfMAX_U8;
    normal = glm::normalize(2.f * glm::vec3(data.normal) * invfMAX_U16 - 1.f);
    ior = data.ior;
  }

  inline SFilteredDataCompact::SFilteredDataCompact(const SFilteredData& data) {
    density = data.density;
    uint16_t MAX_U16 = -1;
    float fMAX_U16 = MAX_U16;
    uint8_t MAX_U8 = -1;
    float fMAX_U8 = MAX_U8;
    sigma_x = glm::clamp(fMAX_U16 * glm::sqrt(data.S[0][0]), 0.f, fMAX_U16);
    sigma_y = glm::clamp(fMAX_U16 * glm::sqrt(data.S[1][1]), 0.f, fMAX_U16);
    sigma_z = glm::clamp(fMAX_U16 * glm::sqrt(data.S[2][2]), 0.f, fMAX_U16);
    r_xy = glm::clamp(fMAX_U16 * 0.5f * (data.S[0][1] / glm::sqrt(data.S[0][0] * data.S[1][1]) + 1.f), 0.f, fMAX_U16);
    r_xz = glm::clamp(fMAX_U16 * 0.5f * (data.S[0][2] / glm::sqrt(data.S[0][0] * data.S[2][2]) + 1.f), 0.f, fMAX_U16);
    r_yz = glm::clamp(fMAX_U16 * 0.5f * (data.S[1][2] / glm::sqrt(data.S[1][1] * data.S[2][2]) + 1.f), 0.f, fMAX_U16);
    diffuseColor = glm::clamp(fMAX_U8 * data.diffuseColor, 0.f, fMAX_U8);
    specularColor = glm::clamp(fMAX_U8 * data.specularColor, 0.f, fMAX_U8);
    normal = glm::clamp(fMAX_U16 * 0.5f * (data.n() + 1.f), 0.f, fMAX_U16);
    ior = data.ior;
  }

  inline glm::vec3 SFilteredData::n() const {
    return glm::normalize(normal);
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
    out.S = glm::mat3(S_xx, S_xy, S_xz,
                      S_xy, S_yy, S_yz,
                      S_xz, S_yz, S_zz);
    out.diffuseColor = v * filteredData.diffuseColor;
    out.specularColor = v * filteredData.specularColor;
    out.normal = v * filteredData.n();
    out.ior = v * filteredData.ior;
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
    out.S = glm::mat3(S_xx, S_xy, S_xz,
                      S_xy, S_yy, S_yz,
                      S_xz, S_yz, S_zz);
    out.diffuseColor = v1.diffuseColor + v2.diffuseColor;
    out.specularColor = v1.specularColor + v2.specularColor;
    out.normal = glm::normalize(v1.normal + v2.normal);
    out.ior = v1.ior + v2.ior;
    return out;
  }
}

#endif // !FILTERED_DATA_HPP
