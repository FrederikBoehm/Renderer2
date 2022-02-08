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

    m_launchParams.numVoxels = config.numVoxels;

    glm::vec3 minModel(FLT_MAX);
    glm::vec3 maxModel(-FLT_MAX);
    for (const auto& bb : config.modelSpaceBoundingBoxes) {
      minModel = glm::min(bb.m_min, minModel);
      maxModel = glm::max(bb.m_max, maxModel);
    }

    glm::mat4 indexToModel = glm::translate(minModel) * glm::scale(maxModel - minModel) * glm::scale(1.f / glm::vec3(config.numVoxels)); // Index space from [0, numVoxels - 1]
    m_launchParams.indexToModel = indexToModel;
    m_launchParams.modelToIndex = glm::inverse(indexToModel);

    openvdb::Mat4R transformMatrix(
      indexToModel[0][0], indexToModel[0][1], indexToModel[0][2], indexToModel[0][3],
      indexToModel[1][0], indexToModel[1][1], indexToModel[1][2], indexToModel[1][3],
      indexToModel[2][0], indexToModel[2][1], indexToModel[2][2], indexToModel[2][3],
      indexToModel[3][0], indexToModel[3][1], indexToModel[3][2], indexToModel[3][3]); // openvdb::Mat4R says row-major layout, but isAffine looks like column-major

    m_data->grid->setTransform(openvdb::math::Transform::createLinearTransform(transformMatrix));

    glm::vec3 minWorld(FLT_MAX);
    glm::vec3 maxWorld(-FLT_MAX);
    for (const auto& bb : config.worldSpaceBoundingBoxes) {
      minWorld = glm::min(bb.m_min, minWorld);
      maxWorld = glm::max(bb.m_max, maxWorld);
    }

    m_launchParams.worldBB = { minWorld, maxWorld };

    glm::mat4 modelToWorld = glm::translate(minWorld) * glm::scale(maxWorld - minWorld) * glm::scale(1.f / (maxModel - minModel)) * glm::translate(-minModel);
    m_launchParams.modelToWorld = modelToWorld;
    m_launchParams.worldToModel = glm::inverse(modelToWorld);
  }

  void COpenvdbBackend::setValues(const std::vector<float>& filteredData) {
    if (!m_data) {
      throw new std::exception("Called COpenvdbBackend::setValues without initializing COpenvdbBackend backend.");
    }

    for (int x = 0; x < m_data->numVoxels.x; ++x) {
      for (int y = 0; y < m_data->numVoxels.y; ++y) {
        for (int z = 0; z < m_data->numVoxels.z; ++z) {
          size_t id = x + y * m_data->numVoxels.x + z * m_data->numVoxels.x * m_data->numVoxels.y;
          m_data->accessor.setValue(openvdb::Coord(x, y, z), filteredData[id]);
        }
      }
    }

  }

  nanovdb::GridHandle<nanovdb::HostBuffer> COpenvdbBackend::getNanoGridHandle() const {
    if (!m_data) {
      throw new std::exception("Called COpenvdbBackend::getNanoGrid without initializing COpenvdbBackend backend.");
    }

    auto handle = nanovdb::openToNanoVDB(m_data->grid);
    auto* dstGrid = handle.grid<float>();
    if (!dstGrid)
      throw std::runtime_error("GridHandle does not contain a grid with value type float");
    return handle;
  }

  void COpenvdbBackend::writeToFile(const nanovdb::GridHandle<nanovdb::HostBuffer>& gridHandle, const char* directory, const char* fileName) const {
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
