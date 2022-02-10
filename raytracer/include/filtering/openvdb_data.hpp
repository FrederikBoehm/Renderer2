#ifndef OPENVDB_DATA_HPP
#define OPENVDB_DATA_HPP
#include <openvdb/openvdb.h>
#include <glm/glm.hpp>
using Vec4DTree = openvdb::tree::Tree4<openvdb::Vec4d, 5, 4, 3>::Type;
using Vec4DGrid = openvdb::Grid<Vec4DTree>;

namespace filter {
  struct SOpenvdbData {
    Vec4DGrid::Ptr grid;
    Vec4DGrid::Accessor accessor;
    glm::ivec3 numVoxels;
  };
}
#endif // !OPENVDB_DATA_HPP
