#ifndef MESH_LOADER_HPP
#define MESH_LOADER_HPP
#include <assimp/Importer.hpp>
#include <assimp/scene.h>     
#include <assimp/postprocess.h>
#include "utility/qualifiers.hpp"
#include <string>
namespace rt {
  class CMeshLoader {
  public:
    H_CALLABLE static const aiScene* loadScene(const std::string& path);
  };
}
#endif