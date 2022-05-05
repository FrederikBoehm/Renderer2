#ifndef VOLUME_DESCRIPTION_MANAGER_HPP
#define VOLUME_DESCRIPTION_MANAGER_HPP
#include <unordered_map>
#include <string>
#include <glm/glm.hpp>


namespace filter {

  struct SVolumeDescription{
    //float voxelSize;
    glm::ivec3 numVoxels;
    glm::vec3 bbmin;
    glm::vec3 bbmax;
    glm::vec3 p0;
    glm::vec3 p1;
    glm::vec3 p2;
    glm::vec3 p3;
    glm::vec3 p4;
    glm::vec3 p5;
    glm::vec3 p6;
    glm::vec3 p7;
  };

  // Exports information about aabb sizes, which is required for scene generation
  class CVolumeDescriptionManager {
  public:
    static CVolumeDescriptionManager* instance();

    void loadDescriptions(const std::string& path);
    void storeDescriptions(const std::string& path) const;
    void addDescription(const std::string& model, float voxelSize, const glm::vec3& bbmin, const glm::vec3& bbmax, const glm::ivec3& numVoxels);

  private:
    static CVolumeDescriptionManager* s_instance;

    std::unordered_map<std::string, std::unordered_map<std::string, SVolumeDescription>> m_descriptions;

    CVolumeDescriptionManager() = default;
    ~CVolumeDescriptionManager() = default;

    CVolumeDescriptionManager(const CVolumeDescriptionManager&) = delete;
    CVolumeDescriptionManager(CVolumeDescriptionManager&&) = delete;
    CVolumeDescriptionManager& operator=(const CVolumeDescriptionManager&) = delete;
    CVolumeDescriptionManager& operator=(const CVolumeDescriptionManager&&) = delete;

    void getBoundingPoints(const glm::vec3& bbmin,
                           const glm::vec3& bbmax,
                           glm::vec3* p0,
                           glm::vec3* p1,
                           glm::vec3* p2,
                           glm::vec3* p3,
                           glm::vec3* p4,
                           glm::vec3* p5,
                           glm::vec3* p6,
                           glm::vec3* p7) const;
  };

  inline CVolumeDescriptionManager* CVolumeDescriptionManager::instance() {
    if (!s_instance) {
      s_instance = new CVolumeDescriptionManager();
    }
    return s_instance;
  }
}

#endif // !VOLUME_DESCRIPTION_MANAGER_HPP
