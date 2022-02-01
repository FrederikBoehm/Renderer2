#ifndef OPENVDB_DATA_HPP
#define OPENVDB_DATA_HPP
#include <openvdb/openvdb.h>
#include <glm/glm.hpp>
namespace filter {
  struct SOpenvdbData {
    openvdb::FloatGrid::Ptr grid;
    openvdb::FloatGrid::Accessor accessor;
    glm::ivec3 numVoxels;
  };
}
#endif // !OPENVDB_DATA_HPP
