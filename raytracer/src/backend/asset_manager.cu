#include "backend/asset_manager.hpp"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/mesh.h>
#include <iostream>
#include "mesh/mesh.hpp"
#include "material/material.hpp"
#include "utility/debugging.hpp"
#include "medium/nvdb_medium.hpp"
#include "texture/texture.hpp"

namespace rt {
  std::unordered_map<std::string, size_t> CAssetManager::s_submeshes;
  std::unordered_map<SMeshKey, CMesh*, SMeshHasher> CAssetManager::s_hostMeshes;
  std::unordered_map<SMeshKey, CMesh*, SMeshHasher> CAssetManager::s_deviceMeshes;
  std::unordered_map<SMaterialKey, CMaterial*, SMaterialHasher> CAssetManager::s_hostMaterials;
  std::unordered_map<SMaterialKey, CMaterial*, SMaterialHasher> CAssetManager::s_deviceMaterials;
  std::unordered_map<SMediumKey, CNVDBMedium*, SMediumHasher> CAssetManager::s_hostMedia;
  std::unordered_map<SMediumKey, CNVDBMedium*, SMediumHasher> CAssetManager::s_deviceMedia;
  std::unordered_map<STextureKey, CTexture*, STextureHasher> CAssetManager::s_hostTextures;
  std::unordered_map<STextureKey, CTexture*, STextureHasher> CAssetManager::s_deviceTextures;

  std::vector<std::tuple<CMesh*, CMaterial*>> CAssetManager::loadWithAssimp(const std::string& assetsBasePath, const std::string& meshFileName, size_t submeshOffset) {
    std::string meshPath = assetsBasePath + "/" + meshFileName;
    
    Assimp::Importer importer;

    const aiScene* scene = importer.ReadFile(meshPath,
      aiProcess_Triangulate | aiProcess_GenNormals | aiProcess_FlipUVs | aiProcess_JoinIdenticalVertices | aiProcess_OptimizeMeshes | aiProcess_GenBoundingBoxes);

    if (!scene) {
      std::cerr << "[ERROR] Failed to load scene: " << importer.GetErrorString() << std::endl;
    }

    std::vector<std::tuple<CMesh*, CMaterial*>> meshes;
    meshes.reserve(scene->mNumMeshes);
    for (unsigned int i = submeshOffset; i < scene->mNumMeshes; ++i) {
      const aiMesh* meshAi = scene->mMeshes[i];
      const aiMaterial* materialAi = scene->mMaterials[meshAi->mMaterialIndex];

      std::vector<glm::vec3> vbo;
      vbo.reserve(meshAi->mNumVertices);
      std::vector<glm::vec3> normals;
      normals.reserve(meshAi->mNumVertices);
      std::vector<glm::vec2> tcs;
      bool hasTcs = meshAi->HasTextureCoords(0);
      if (hasTcs) {
        tcs.reserve(meshAi->mNumVertices);
      }
      for (unsigned int i = 0; i < meshAi->mNumVertices; ++i) {
        const aiVector3D& vertex = meshAi->mVertices[i];
        vbo.emplace_back(vertex.x, vertex.y, vertex.z);
        const aiVector3D& normal = meshAi->mNormals[i];
        normals.emplace_back(normal.x, normal.y, normal.z);
        if (hasTcs) {
          const aiVector3D &tc = meshAi->mTextureCoords[0][i];
          tcs.emplace_back(tc.x, tc.y);
        }
      }

      std::vector<glm::uvec3> ibo;
      ibo.reserve(meshAi->mNumFaces);
      for (unsigned int i = 0; i < meshAi->mNumFaces; ++i) {
        const aiFace& face = meshAi->mFaces[i];
        ibo.emplace_back(face.mIndices[0], face.mIndices[1], face.mIndices[2]);
      }

      CMesh* mesh = new CMesh(meshPath, i, vbo, ibo, normals, tcs, reinterpret_cast<const SAABB&>(meshAi->mAABB));
      CMaterial* material = new CMaterial(materialAi, assetsBasePath, i);

      s_hostMeshes[{meshPath, i}] = mesh;
      s_hostMaterials[{assetsBasePath, i}] = material;


      meshes.emplace_back(mesh, material);

    }

    s_submeshes[meshPath] = scene->mNumMeshes;

    return meshes;
  }

  std::vector<std::tuple<CMesh*, CMaterial*>> CAssetManager::loadMesh(const std::string& assetsBasePath, const std::string& meshFileName) {
    std::string meshPath = assetsBasePath + "/" + meshFileName;
    auto submeshIter = s_submeshes.find(meshPath);
    std::vector<std::tuple<CMesh*, CMaterial*>> meshes;
    if (submeshIter != s_submeshes.end()) {
      for (unsigned int i = 0; i < submeshIter->second; ++i) {
        auto meshIter = s_hostMeshes.find({ meshPath, i });
        auto materialIter = s_hostMaterials.find({ assetsBasePath, i });
        if (meshIter != s_hostMeshes.end() && materialIter != s_hostMaterials.end()) {
          meshes.emplace_back(meshIter->second, materialIter->second);
        }
        else {
          // Continue with loading file
          std::vector<std::tuple<CMesh*, CMaterial*>> loaded = loadWithAssimp(assetsBasePath, meshFileName, i);
          meshes.insert(meshes.end(), loaded.begin(), loaded.end());
          break;
        }
      }
    }
    else {
      std::vector<std::tuple<CMesh*, CMaterial*>> loaded = loadWithAssimp(assetsBasePath, meshFileName);
      meshes.insert(meshes.end(), loaded.begin(), loaded.end());
    }

    return meshes;

  }

  CNVDBMedium* CAssetManager::loadMedium(const std::string& path, const glm::vec3& sigma_a, const glm::vec3& sigma_s, float diffuseRoughness, float specularRoughness) {
    CNVDBMedium* medium = nullptr;
    auto mediumIter = s_hostMedia.find({ path });
    if (mediumIter != s_hostMedia.end()) {
      return mediumIter->second;
    }
    else {
      medium = new CNVDBMedium(path, sigma_a, sigma_s, diffuseRoughness, specularRoughness);
      s_hostMedia[{path}] = medium;
    }
    return medium;
  }

  CNVDBMedium* CAssetManager::loadMedium(const std::string& path, const glm::vec3& sigma_a, const glm::vec3& sigma_s, float g) {
    CNVDBMedium* medium = nullptr;
    auto mediumIter = s_hostMedia.find({ path });
    if (mediumIter != s_hostMedia.end()) {
      return mediumIter->second;
    }
    else {
      medium = new CNVDBMedium(path, sigma_a, sigma_s, g);
      s_hostMedia[{path}] = medium;
    }
    return medium;
  }

  CTexture* CAssetManager::loadTexture(const std::string& path, ETextureType type) {
    auto texIter = s_hostTextures.find({ path, type });
    if (texIter != s_hostTextures.end()) {
      return texIter->second;
    }
    else {
      CTexture* tex = new CTexture(path, type);
      s_hostTextures[{path, type}] = tex;
      return tex;
    }
  }

  CTexture* CAssetManager::loadAlpha(const std::string& path) {
    auto texIter = s_hostTextures.find({ path, ALPHA });
    if (texIter != s_hostTextures.end()) {
      return texIter->second;
    }
    else {
      CTexture* tex = new CTexture();
      tex->loadAlpha(path);
      s_hostTextures[{path, ALPHA}] = tex;
      return tex;
    }
  }

  void CAssetManager::allocateDeviceMemory() {
    for (auto tex : s_hostTextures) {
      CUDA_ASSERT(cudaMalloc(&s_deviceTextures[tex.first], sizeof(CTexture)));
      tex.second->allocateDeviceMemory();
    }
    for (auto mesh : s_hostMeshes) {
      CUDA_ASSERT(cudaMalloc(&s_deviceMeshes[mesh.first], sizeof(CMesh)));
      mesh.second->allocateDeviceMemory();
    }
    for (auto material : s_hostMaterials) {
      CUDA_ASSERT(cudaMalloc(&s_deviceMaterials[material.first], sizeof(CMaterial)));
      material.second->allocateDeviceMemory();
    }
    for (auto medium : s_hostMedia) {
      CUDA_ASSERT(cudaMalloc(&s_deviceMedia[medium.first], sizeof(CNVDBMedium)));
      medium.second->allocateDeviceMemory();
    }
  }

  void CAssetManager::copyToDevice() {
    for (auto tex : s_hostTextures) {
      CUDA_ASSERT(cudaMemcpy(s_deviceTextures[tex.first], &tex.second->copyToDevice(), sizeof(CTexture), cudaMemcpyHostToDevice));
    }
    for (auto mesh : s_hostMeshes) {
      CUDA_ASSERT(cudaMemcpy(s_deviceMeshes[mesh.first], &mesh.second->copyToDevice(), sizeof(CMesh), cudaMemcpyHostToDevice));
    }
    for (auto material : s_hostMaterials) {
      CUDA_ASSERT(cudaMemcpy(s_deviceMaterials[material.first], &material.second->copyToDevice(), sizeof(CMaterial), cudaMemcpyHostToDevice));
    }
    for (auto medium : s_hostMedia) {
      CUDA_ASSERT(cudaMemcpy(s_deviceMedia[medium.first], &medium.second->copyToDevice(), sizeof(CNVDBMedium), cudaMemcpyHostToDevice));
    }
  }

  void CAssetManager::freeDeviceMemory() {
    for (auto tex : s_deviceTextures) {
      s_hostTextures[tex.first]->freeDeviceMemory();
      CUDA_ASSERT(cudaFree(tex.second));
    }
    for (auto mesh : s_deviceMeshes) {
      s_hostMeshes[mesh.first]->freeDeviceMemory();
      CUDA_ASSERT(cudaFree(mesh.second));
    }
    for (auto material : s_deviceMaterials) {
      s_hostMaterials[material.first]->freeDeviceMemory();
      CUDA_ASSERT(cudaFree(material.second));
    }
    for (auto medium : s_deviceMedia) {
      s_hostMedia[medium.first]->freeDeviceMemory();
      CUDA_ASSERT(cudaFree(medium.second));
    }
  }

  void CAssetManager::buildOptixAccel() {
    for (auto mesh : s_hostMeshes) {
      mesh.second->buildOptixAccel();
    }
    for (auto medium : s_hostMedia) {
      medium.second->buildOptixAccel();
    }
  }

  CMesh* CAssetManager::deviceMesh(const std::string& path, size_t submeshId) {
    auto meshIter = s_deviceMeshes.find({ path, submeshId });
    if (meshIter != s_deviceMeshes.end()) {
      return meshIter->second;
    }
    else {
      return nullptr;
    }
  }

  CMaterial* CAssetManager::deviceMaterial(const std::string& path, size_t submeshId) {
    auto materialIter = s_deviceMaterials.find({ path, submeshId });
    if (materialIter != s_deviceMaterials.end()) {
      return materialIter->second;
    }
    else {
      return nullptr;
    }
  }

  CNVDBMedium* CAssetManager::deviceMedium(const std::string& path) {
    auto mediumIter = s_deviceMedia.find({ path });
    if (mediumIter != s_deviceMedia.end()) {
      return mediumIter->second;
    }
    else {
      return nullptr;
    }
  }

  CTexture* CAssetManager::deviceTexture(const std::string& path, ETextureType type) {
    auto texIter = s_deviceTextures.find({ path, type });
    if (texIter != s_deviceTextures.end()) {
      return texIter->second;
    }
    else {
      return nullptr;
    }
  }

  void CAssetManager::release() {
    for (auto texture : s_hostTextures) {
      delete texture.second;
    }
    for (auto mesh : s_hostMeshes) {
      delete mesh.second;
    }
    for (auto medium : s_hostMedia) {
      delete medium.second;
    }
    for (auto material : s_hostMaterials) {
      delete material.second;
    }
  }
}