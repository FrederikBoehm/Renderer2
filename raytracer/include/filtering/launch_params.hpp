#ifndef FILTER_LAUNCH_PARAMS_HPP
#define FILTER_LAUNCH_PARAMS_HPP
#include <glm/glm.hpp>
#include "intersect/aabb.hpp"

namespace rt {
  class CSampler;
  class CDeviceScene;
}

namespace filter {
  struct SFilteredData;

  struct SFilterLaunchParams {
    glm::mat4x3 indexToModel;
    glm::mat4x3 modelToIndex;
    glm::mat4x3 modelToWorld;
    glm::mat4x3 worldToModel;
    glm::ivec3 numVoxels;
    rt::SAABB worldBB;
    rt::CSampler* samplers;
    rt::CDeviceScene* scene;
    uint32_t samplesPerVoxel;
    SFilteredData* filteredData;
    bool debug;
    uint32_t debugSamples;
    float sigma_t;
    uint32_t estimationIterations;
    float alpha;
    bool clipRays;
  };
}

#endif