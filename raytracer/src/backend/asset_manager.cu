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

namespace rt {
  std::unordered_map<SMeshKey, CMesh*, SMeshHasher> CAssetManager::s_hostMeshes;
  std::unordered_map<SMeshKey, CMesh*, SMeshHasher> CAssetManager::s_deviceMeshes;
  std::unordered_map<SMediumKey, CNVDBMedium*, SMediumHasher> CAssetManager::s_hostMedia;
  std::unordered_map<SMediumKey, CNVDBMedium*, SMediumHasher> CAssetManager::s_deviceMedia;

  std::vector<std::tuple<CMesh*, CMaterial*>> CAssetManager::loadMesh(const std::string& assetsBasePath, const std::string& meshFileName) {
    Assimp::Importer importer;

    const aiScene* scene = importer.ReadFile(assetsBasePath + "/" + meshFileName,
      aiProcess_Triangulate | aiProcess_GenNormals | aiProcess_FlipUVs | aiProcess_JoinIdenticalVertices | aiProcess_OptimizeMeshes);

    if (!scene) {
      std::cerr << "[ERROR] Failed to load scene: " << importer.GetErrorString() << std::endl;
    }

    std::vector<std::tuple<CMesh*, CMaterial*>> meshes;
    meshes.reserve(scene->mNumMeshes);
    for (unsigned int i = 0; i < scene->mNumMeshes; ++i) {
      const aiMesh* meshAi = scene->mMeshes[i];
      const aiMaterial* material = scene->mMaterials[meshAi->mMaterialIndex];

      std::string meshPath = assetsBasePath + "/" + meshFileName;
      aiAABB aabb = meshAi->mAABB;
      OptixAabb aabbOptix;
      sizeof(aiAABB);
      sizeof(OptixAabb);
      sizeof(nanovdb::BBoxR);

      CMesh* mesh = nullptr;
      auto meshIter = s_hostMeshes.find({ meshPath, i });
      if (meshIter != s_hostMeshes.end()) {
        mesh = meshIter->second;
      }
      else {
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

        mesh = new CMesh(meshPath, i, vbo, ibo, normals, tcs);

        s_hostMeshes[{meshPath, i}] = mesh;

      }

      meshes.emplace_back(mesh, new CMaterial(material, assetsBasePath));

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

  void CAssetManager::allocateDeviceMemory() {
    for (auto mesh : s_hostMeshes) {
      CUDA_ASSERT(cudaMalloc(&s_deviceMeshes[mesh.first], sizeof(CMesh)));
      mesh.second->allocateDeviceMemory();
    }
    for (auto medium : s_hostMedia) {
      CUDA_ASSERT(cudaMalloc(&s_deviceMedia[medium.first], sizeof(CNVDBMedium)));
      medium.second->allocateDeviceMemory();
    }
  }

  void CAssetManager::copyToDevice() {
    for (auto mesh : s_hostMeshes) {
      CUDA_ASSERT(cudaMemcpy(s_deviceMeshes[mesh.first], &mesh.second->copyToDevice(), sizeof(CMesh), cudaMemcpyHostToDevice));
    }
    for (auto medium : s_hostMedia) {
      CUDA_ASSERT(cudaMemcpy(s_deviceMedia[medium.first], &medium.second->copyToDevice(), sizeof(CNVDBMedium), cudaMemcpyHostToDevice));
    }
  }

  void CAssetManager::freeDeviceMemory() {
    for (auto mesh : s_deviceMeshes) {
      s_hostMeshes[mesh.first]->freeDeviceMemory();
      CUDA_ASSERT(cudaFree(mesh.second));
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

  CNVDBMedium* CAssetManager::deviceMedium(const std::string& path) {
    auto mediumIter = s_deviceMedia.find({ path });
    if (mediumIter != s_deviceMedia.end()) {
      return mediumIter->second;
    }
    else {
      return nullptr;
    }
  }

  void CAssetManager::release() {
    for (auto mesh : s_hostMeshes) {
      delete mesh.second;
    }
    for (auto medium : s_hostMedia) {
      delete medium.second;
    }
  }
}