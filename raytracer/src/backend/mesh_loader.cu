#include "backend/mesh_loader.hpp"
#include <iostream>

namespace rt {
  const aiScene* CMeshLoader::loadScene(const std::string& path) {
    Assimp::Importer importer;

    const aiScene* scene = importer.ReadFile(path,
      aiProcess_Triangulate | aiProcess_GenNormals | aiProcess_FlipUVs | aiProcess_JoinIdenticalVertices | aiProcess_OptimizeMeshes);

    if (!scene) {
      std::cerr << "[ERROR] Failed to load scene: " << importer.GetErrorString() << std::endl;
    }

    return scene;
  }
}