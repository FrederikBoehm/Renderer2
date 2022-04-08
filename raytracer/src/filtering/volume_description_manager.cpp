#include "filtering/volume_description_manager.hpp"
#include <json/json.hpp>
#include <fstream>
#include <filesystem>

namespace filter {
  CVolumeDescriptionManager* CVolumeDescriptionManager::s_instance = nullptr;


  void CVolumeDescriptionManager::loadDescriptions(const std::string& path) {
    if (std::filesystem::exists(path)) {
      std::ifstream inputStream(path);
      std::stringstream buffer;
      buffer << inputStream.rdbuf();
      std::string jsonString = buffer.str();
      nlohmann::json jDescription = nlohmann::json::parse(jsonString, nullptr, true, true);

      for (nlohmann::json::iterator itModel = jDescription.begin(); itModel != jDescription.end(); ++itModel) {
        for (nlohmann::json::iterator itVoxelSize = itModel.value().begin(); itVoxelSize != itModel.value().end(); ++itVoxelSize) {
          SVolumeDescription volumeDescription;
          //volumeDescription.voxelSize = it.value()["VoxelSize"].get<float>();
          const nlohmann::json& numVoxels = itVoxelSize.value()["NumVoxels"];
          volumeDescription.numVoxels = glm::ivec3(numVoxels[0].get<int>(), numVoxels[1].get<int>(), numVoxels[2].get<int>());
          const nlohmann::json& bbmin = itVoxelSize.value()["BBMin"];
          volumeDescription.bbmin = glm::vec3(bbmin[0].get<float>(), bbmin[1].get<float>(), bbmin[2].get<float>());
          const nlohmann::json& bbmax = itVoxelSize.value()["BBMax"];
          volumeDescription.bbmax = glm::vec3(bbmax[0].get<float>(), bbmax[1].get<float>(), bbmax[2].get<float>());
          const nlohmann::json& p0 = itVoxelSize.value()["P0"];
          volumeDescription.p0 = glm::vec3(p0[0].get<float>(), p0[1].get<float>(), p0[2].get<float>());
          const nlohmann::json& p1 = itVoxelSize.value()["P1"];
          volumeDescription.p1 = glm::vec3(p1[0].get<float>(), p1[1].get<float>(), p1[2].get<float>());
          const nlohmann::json& p2 = itVoxelSize.value()["P2"];
          volumeDescription.p2 = glm::vec3(p2[0].get<float>(), p2[1].get<float>(), p2[2].get<float>());
          const nlohmann::json& p3 = itVoxelSize.value()["P3"];
          volumeDescription.p3 = glm::vec3(p3[0].get<float>(), p3[1].get<float>(), p3[2].get<float>());
          const nlohmann::json& p4 = itVoxelSize.value()["P4"];
          volumeDescription.p4 = glm::vec3(p4[0].get<float>(), p4[1].get<float>(), p4[2].get<float>());
          const nlohmann::json& p5 = itVoxelSize.value()["P5"];
          volumeDescription.p5 = glm::vec3(p5[0].get<float>(), p5[1].get<float>(), p5[2].get<float>());
          const nlohmann::json& p6 = itVoxelSize.value()["P6"];
          volumeDescription.p6 = glm::vec3(p6[0].get<float>(), p6[1].get<float>(), p6[2].get<float>());
          const nlohmann::json& p7 = itVoxelSize.value()["P7"];
          volumeDescription.p7 = glm::vec3(p7[0].get<float>(), p7[1].get<float>(), p7[2].get<float>());
          m_descriptions[itModel.key()][itVoxelSize.key()] = volumeDescription;
        }
      }
    }
  }

  void CVolumeDescriptionManager::storeDescriptions(const std::string& path) const {
    nlohmann::json jDescription;
    for (auto itModel = m_descriptions.begin(); itModel != m_descriptions.end(); ++itModel) {
      for (auto itVoxelSize = itModel->second.begin(); itVoxelSize != itModel->second.end(); ++itVoxelSize) {
        nlohmann::json& numVoxels = jDescription[itModel->first][itVoxelSize->first]["NumVoxels"];
        numVoxels[0] = itVoxelSize->second.numVoxels.x;
        numVoxels[1] = itVoxelSize->second.numVoxels.y;
        numVoxels[2] = itVoxelSize->second.numVoxels.z;

        nlohmann::json& bbmin = jDescription[itModel->first][itVoxelSize->first]["BBMin"];
        bbmin[0] = itVoxelSize->second.bbmin.x;
        bbmin[1] = itVoxelSize->second.bbmin.y;
        bbmin[2] = itVoxelSize->second.bbmin.z;

        nlohmann::json& bbmax = jDescription[itModel->first][itVoxelSize->first]["BBMax"];
        bbmax[0] = itVoxelSize->second.bbmax.x;
        bbmax[1] = itVoxelSize->second.bbmax.y;
        bbmax[2] = itVoxelSize->second.bbmax.z;

        nlohmann::json& p0 = jDescription[itModel->first][itVoxelSize->first]["P0"];
        p0[0] = itVoxelSize->second.p0.x;
        p0[1] = itVoxelSize->second.p0.y;
        p0[2] = itVoxelSize->second.p0.z;

        nlohmann::json& p1 = jDescription[itModel->first][itVoxelSize->first]["P1"];
        p1[0] = itVoxelSize->second.p1.x;
        p1[1] = itVoxelSize->second.p1.y;
        p1[2] = itVoxelSize->second.p1.z;

        nlohmann::json& p2 = jDescription[itModel->first][itVoxelSize->first]["P2"];
        p2[0] = itVoxelSize->second.p2.x;
        p2[1] = itVoxelSize->second.p2.y;
        p2[2] = itVoxelSize->second.p2.z;

        nlohmann::json& p3 = jDescription[itModel->first][itVoxelSize->first]["P3"];
        p3[0] = itVoxelSize->second.p3.x;
        p3[1] = itVoxelSize->second.p3.y;
        p3[2] = itVoxelSize->second.p3.z;

        nlohmann::json& p4 = jDescription[itModel->first][itVoxelSize->first]["P4"];
        p4[0] = itVoxelSize->second.p4.x;
        p4[1] = itVoxelSize->second.p4.y;
        p4[2] = itVoxelSize->second.p4.z;

        nlohmann::json& p5 = jDescription[itModel->first][itVoxelSize->first]["P5"];
        p5[0] = itVoxelSize->second.p5.x;
        p5[1] = itVoxelSize->second.p5.y;
        p5[2] = itVoxelSize->second.p5.z;

        nlohmann::json& p6 = jDescription[itModel->first][itVoxelSize->first]["P6"];
        p6[0] = itVoxelSize->second.p6.x;
        p6[1] = itVoxelSize->second.p6.y;
        p6[2] = itVoxelSize->second.p6.z;

        nlohmann::json& p7 = jDescription[itModel->first][itVoxelSize->first]["P7"];
        p7[0] = itVoxelSize->second.p7.x;
        p7[1] = itVoxelSize->second.p7.y;
        p7[2] = itVoxelSize->second.p7.z;
      }
    }

    std::string jsonString = jDescription.dump(2);
    std::ofstream outputStream(path);
    outputStream << jsonString;
  }

  void CVolumeDescriptionManager::addDescription(const std::string& model, float voxelSize, const glm::vec3& bbmin, const glm::vec3& bbmax, const glm::ivec3& numVoxels) {
    SVolumeDescription volumeDescription;
    volumeDescription.bbmin = bbmin;
    volumeDescription.bbmax = bbmax;
    volumeDescription.numVoxels = numVoxels;
    getBoundingPoints(bbmin, bbmax, &volumeDescription.p0, &volumeDescription.p1, &volumeDescription.p2, &volumeDescription.p3, &volumeDescription.p4, &volumeDescription.p5, &volumeDescription.p6, &volumeDescription.p7);
    m_descriptions[model][std::to_string(voxelSize)] = volumeDescription;
  }

  void CVolumeDescriptionManager::getBoundingPoints(const glm::vec3& bbmin,
                                                    const glm::vec3& bbmax,
                                                    glm::vec3* p0,
                                                    glm::vec3* p1,
                                                    glm::vec3* p2,
                                                    glm::vec3* p3,
                                                    glm::vec3* p4,
                                                    glm::vec3* p5,
                                                    glm::vec3* p6,
                                                    glm::vec3* p7) const {

    *p0 = bbmin;
    *p1 = glm::vec3(bbmax.x, bbmin.y, bbmin.z);
    *p2 = glm::vec3(bbmax.x, bbmin.y, bbmax.z);
    *p3 = glm::vec3(bbmin.x, bbmin.y, bbmax.z);
    *p4 = glm::vec3(bbmin.x, bbmax.y, bbmin.z);
    *p5 = glm::vec3(bbmax.x, bbmax.y, bbmin.z);
    *p6 = bbmax;
    *p7 = glm::vec3(bbmin.x, bbmax.y, bbmax.z);
  }
}