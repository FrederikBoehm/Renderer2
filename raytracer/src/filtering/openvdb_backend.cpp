#include "filtering/openvdb_backend.hpp"
#include "filtering/openvdb_data.hpp"
#include <exception>
#include <glm/gtx/transform.hpp>
#include <nanovdb/util/OpenToNanoVDB.h>
#include <nanovdb/util/IO.h>
#include <filesystem>

namespace filter {
  COpenvdbBackend* COpenvdbBackend::s_instance = nullptr;

  COpenvdbBackend::COpenvdbBackend() :
    m_data(nullptr) {

  }

  COpenvdbBackend::~COpenvdbBackend() {
    delete m_data;
  }

  void COpenvdbBackend::init(const SOpenvdbBackendConfig& config) {
    if (!m_data) {
      openvdb::initialize();
      openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create();
      openvdb::FloatGrid::Accessor accessor = grid->getAccessor();
      m_data = new SOpenvdbData{ grid, accessor, config.numVoxels };
    }

    glm::vec3 min(FLT_MAX);
    glm::vec3 max(-FLT_MAX);
    for (const auto& bb : config.boundingBoxes) {
      min = glm::min(bb.m_min, min);
      max = glm::max(bb.m_max, max);
    }

    glm::mat4 indexToWorld = glm::translate(min) * glm::scale(max - min) * glm::scale(1.f / glm::vec3(config.numVoxels));
    
    openvdb::Mat4R transformMatrix(
      indexToWorld[0][0], indexToWorld[0][1], indexToWorld[0][2], indexToWorld[0][3],
      indexToWorld[1][0], indexToWorld[1][1], indexToWorld[1][2], indexToWorld[1][3],
      indexToWorld[2][0], indexToWorld[2][1], indexToWorld[2][2], indexToWorld[2][3],
      indexToWorld[3][0], indexToWorld[3][1], indexToWorld[3][2], indexToWorld[3][3]); // openvdb::Mat4R says row-major layout, but isAffine looks like column-major
    
    m_data->grid->setTransform(openvdb::math::Transform::createLinearTransform(transformMatrix));
  }

  void COpenvdbBackend::setValues() {
    if (!m_data) {
      throw new std::exception("Called COpenvdbBackend::setValues without initializing COpenvdbBackend backend.");
    }

    for (int x = 0; x < m_data->numVoxels.x; ++x) {
      for (int y = 0; y < m_data->numVoxels.y; ++y) {
        for (int z = 0; z < m_data->numVoxels.z; ++z) {
          m_data->accessor.setValue(openvdb::Coord(x, y, z), 1.f); // For now, fill grid only with 1
        }
      }
    }

  }

  nanovdb::GridHandle<nanovdb::HostBuffer> COpenvdbBackend::getNanoGridHandle() {
    if (!m_data) {
      throw new std::exception("Called COpenvdbBackend::getNanoGrid without initializing COpenvdbBackend backend.");
    }

    auto handle = nanovdb::openToNanoVDB(m_data->grid);
    auto* dstGrid = handle.grid<float>();
    if (!dstGrid)
      throw std::runtime_error("GridHandle does not contain a grid with value type float");
    return handle;
  }

  void COpenvdbBackend::writeToFile(const nanovdb::GridHandle<nanovdb::HostBuffer>& gridHandle, const char* directory, const char* fileName) {
    auto* dstGrid = gridHandle.grid<float>();
    auto iBB = dstGrid->indexBBox();
    auto worldBB = dstGrid->worldBBox();
    std::filesystem::path p(directory);
    std::filesystem::create_directory(p);
    p.append(fileName);
    nanovdb::io::writeGrid(p.string(), gridHandle);
  }

  COpenvdbBackend* COpenvdbBackend::instance() {
    if (!s_instance) {
      s_instance = new COpenvdbBackend();
    }
    return s_instance;
  }

}