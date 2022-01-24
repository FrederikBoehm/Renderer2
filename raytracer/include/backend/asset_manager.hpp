#ifndef ASSET_MANAGER_HPP
#define ASSET_MANAGER_HPP

#include <vector>
#include <tuple>
#include <glm/glm.hpp>
#include <unordered_map>

namespace rt {
  class CMesh;
  class CMaterial;
  class CNVDBMedium;

  struct SMeshKey {
    std::string m_path;
    size_t m_submeshId;

  };

  struct SMeshHasher {
    std::size_t operator()(const SMeshKey& meshKey) const noexcept {
      std::size_t h1 = std::hash<std::string>{}(meshKey.m_path);
      std::size_t h2 = std::hash<size_t>{}(meshKey.m_submeshId);
      return h1 ^ (h2 << 1);
    }

  };

  inline bool operator==(const SMeshKey& l, const SMeshKey& r) {
    return l.m_path == r.m_path && l.m_submeshId == r.m_submeshId;
  }

  inline bool operator!=(const SMeshKey& l, const SMeshKey& r) {
    return !(l == r);
  }

  struct SMediumKey {
    std::string m_path;
  };

  struct SMediumHasher {
    std::size_t operator()(const SMediumKey& mediumKey) const noexcept {
      return std::hash<std::string>{}(mediumKey.m_path);
    }
  };

  inline bool operator==(const SMediumKey& l, const SMediumKey& r) {
    return l.m_path == r.m_path;
  }

  inline bool operator!=(const SMediumKey& l, const SMediumKey& r) {
    return !(l == r);
  }

  struct SMaterialKey {
    std::string m_path;
    size_t m_submeshId;
  };

  struct SMaterialHasher {
    std::size_t operator()(const SMaterialKey& materialKey) const noexcept {
      std::size_t h1 = std::hash<std::string>{}(materialKey.m_path);
      std::size_t h2 = std::hash<size_t>{}(materialKey.m_submeshId);
      return h1 ^ (h2 << 1);
    }
  };

  inline bool operator==(const SMaterialKey& l, const SMaterialKey& r) {
    return l.m_path == r.m_path && l.m_submeshId == r.m_submeshId;
  }

  inline bool operator!=(const SMaterialKey& l, const SMaterialKey& r) {
    return !(l == r);
  }

  class CAssetManager {
  public:
    static std::vector<std::tuple<CMesh*, CMaterial*>> loadMesh(const std::string& assetsBasePath, const std::string& meshFileName);
    static CNVDBMedium* loadMedium(const std::string& path, const glm::vec3& sigma_a, const glm::vec3& sigma_s, float diffuseRoughness, float specularRoughness);
    static void allocateDeviceMemory();
    static void copyToDevice();
    static void freeDeviceMemory();
    static void buildOptixAccel();
    static CMesh* deviceMesh(const std::string& path, size_t submeshId);
    static CMaterial* deviceMaterial(const std::string& path, size_t submeshId);
    static CNVDBMedium* deviceMedium(const std::string& path);
    static void release();

  private:
    static std::unordered_map<std::string, size_t> s_submeshes;
    static std::unordered_map<SMeshKey, CMesh*, SMeshHasher> s_hostMeshes;
    static std::unordered_map<SMeshKey, CMesh*, SMeshHasher> s_deviceMeshes;
    static std::unordered_map<SMaterialKey, CMaterial*, SMaterialHasher> s_hostMaterials;
    static std::unordered_map<SMaterialKey, CMaterial*, SMaterialHasher> s_deviceMaterials;
    static std::unordered_map<SMediumKey, CNVDBMedium*, SMediumHasher> s_hostMedia;
    static std::unordered_map<SMediumKey, CNVDBMedium*, SMediumHasher> s_deviceMedia;

    static std::vector<std::tuple<CMesh*, CMaterial*>> loadWithAssimp(const std::string& assetsBasePath, const std::string& meshFileName, size_t submeshOffset = 0);
  };
}

#endif