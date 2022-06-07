#ifndef SCENE_TYPES_HPP
#define SCENE_TYPES_HPP
#include <optix_types.h>
#include <vector>
namespace rt {
  // Wraps flags for OptixBuildInput. Pointers become invalid if SBuildInputWrapper leaves scope
  struct SBuildInputWrapper {
    OptixBuildInput buildInput = {};
    std::vector<unsigned int> flags;
  };
}

#endif // !SCENE_TYPES_HPP
