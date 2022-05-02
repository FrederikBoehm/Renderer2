#include "filtering/openvdb_backend.hpp"
#include "filtering/openvdb_data.hpp"
#include <exception>
#include <glm/gtx/transform.hpp>
#include <nanovdb/util/OpenToNanoVDB.h>
#include <nanovdb/util/IO.h>
#include <filesystem>
#include "filtering/filtered_data.hpp"
#include "grid_brick/host_grid_brick.hpp"
#include "grid_brick/serialization.hpp"

#define CEIL_EPS 0.001f

namespace filter {
  COpenvdbBackend* COpenvdbBackend::s_instance = nullptr;

  COpenvdbBackend::COpenvdbBackend() :
    m_data(nullptr) {

  }

  COpenvdbBackend::~COpenvdbBackend() {
    delete m_data;
  }

  void COpenvdbBackend::init(const SOpenvdbBackendConfig& config) {
    static_assert(sizeof(SFilteredDataCompact) <= sizeof(openvdb::Vec4d), "SFilteredDataCompact is too large for openvdb::Vec4d");
    if (!m_data) {
      openvdb::initialize();
      SFilteredDataCompact backgroundValue;
      backgroundValue.density = 0.f;
      backgroundValue.sigma_x = 0;
      backgroundValue.sigma_y = 0;
      backgroundValue.sigma_z = 0;
      backgroundValue.r_xy = 0;
      backgroundValue.r_xz = 0;
      backgroundValue.r_yz = 0;
      backgroundValue.diffuseColor = glm::uvec3(0);
      backgroundValue.specularColor = glm::uvec3(0);
      backgroundValue.normal = glm::vec3(0.f);
      backgroundValue.ior = 1.f;
      Vec4DGrid::Ptr grid = Vec4DGrid::create(reinterpret_cast<openvdb::Vec4d&>(backgroundValue));
      Vec4DGrid::Accessor accessor = grid->getAccessor();
      m_data = new SOpenvdbData{ grid, accessor };
    }

    m_minModel = glm::vec3(FLT_MAX);
    m_maxModel = glm::vec3(-FLT_MAX);
    for (const auto& bb : config.modelSpaceBoundingBoxes) {
      m_minModel = glm::min(bb.m_min, m_minModel);
      m_maxModel = glm::max(bb.m_max, m_maxModel);
    }

    m_minWorld = glm::vec3(FLT_MAX);
    m_maxWorld = glm::vec3(-FLT_MAX);
    for (const auto& bb : config.worldSpaceBoundingBoxes) {
      m_minWorld = glm::min(bb.m_min, m_minWorld);
      m_maxWorld = glm::max(bb.m_max, m_maxWorld);
    }

    m_worldToModel = config.worldToModel;

    glm::vec3 fNumVoxels = (m_maxWorld - m_minWorld) / config.voxelSize;
    m_numVoxelsMajorant = glm::ceil(fNumVoxels - glm::vec3(CEIL_EPS));

    glm::vec3 dimensions = glm::abs(glm::inverse(glm::mat4(m_worldToModel)) * glm::vec4(m_maxModel - m_minModel, 0.f));
    printf("World space dimensions: (%f, %f, %f)\n", dimensions.x, dimensions.y, dimensions.z);
    
  }

  SFilterLaunchParams COpenvdbBackend::setupGrid(const glm::vec3& voxelSize) {
    if (!m_data) {
      throw new std::exception("Called COpenvdbBackend::setupGrid without initializing COpenvdbBackend backend.");
    }

    m_data->grid->clear();

    glm::vec3 scaling = 1.f / (m_maxModel - m_minModel);

    glm::vec3 fNumVoxels = (m_maxWorld - m_minWorld) / voxelSize;
    glm::ivec3 numVoxels = glm::ceil(fNumVoxels - glm::vec3(CEIL_EPS));
    SFilterLaunchParams launchParams;
    launchParams.numVoxels = numVoxels;

    glm::mat4 indexToModel = glm::mat4(m_worldToModel) * glm::translate(m_minWorld) * glm::scale(m_maxWorld - m_minWorld) * glm::scale(1.f / glm::vec3(numVoxels));
    launchParams.indexToModel = indexToModel;
    launchParams.modelToIndex = glm::inverse(indexToModel);

    openvdb::Mat4R transformMatrix(
      indexToModel[0][0], indexToModel[0][1], indexToModel[0][2], indexToModel[0][3],
      indexToModel[1][0], indexToModel[1][1], indexToModel[1][2], indexToModel[1][3],
      indexToModel[2][0], indexToModel[2][1], indexToModel[2][2], indexToModel[2][3],
      indexToModel[3][0], indexToModel[3][1], indexToModel[3][2], indexToModel[3][3]); // openvdb::Mat4R says row-major layout, but isAffine looks like column-major

    m_data->grid->setTransform(openvdb::math::Transform::createLinearTransform(transformMatrix));

    glm::mat4 modelToWorld = glm::inverse(glm::mat4(m_worldToModel));
    launchParams.modelToWorld = modelToWorld;
    launchParams.worldToModel = m_worldToModel;
    glm::mat4 indexToWorld = modelToWorld * indexToModel;

    glm::vec3 modelMin = indexToWorld * glm::vec4(0.f, 0.f, 0.f, 1.f);
    glm::vec3 modelMax = indexToWorld * glm::vec4(numVoxels, 1.f);
    launchParams.worldBB = { glm::min(modelMin, modelMax), glm::max(modelMin, modelMax) };
    return launchParams;
  }

  void COpenvdbBackend::setValues(const std::vector<SFilteredDataCompact>& filteredData, const glm::ivec3& numVoxels) {
    if (!m_data) {
      throw new std::exception("Called COpenvdbBackend::setValues without initializing COpenvdbBackend backend.");
    }

    for (int x = 0; x < numVoxels.x; ++x) {
      for (int y = 0; y < numVoxels.y; ++y) {
        for (int z = 0; z < numVoxels.z; ++z) {
          size_t id = x + y * numVoxels.x + z * numVoxels.x * numVoxels.y;
          glm::ivec3 coord(x, y, z);
          if (filteredData[id].density > 0.f ||
              coord == glm::ivec3(0, 0, 0) ||
              coord == glm::ivec3(0, 0, numVoxels.z - 1) ||
              coord == glm::ivec3(0, numVoxels.y - 1, 0) ||
              coord == glm::ivec3(0, numVoxels.y - 1, numVoxels.z - 1) ||
              coord == glm::ivec3(numVoxels.x - 1, 0, 0) ||
              coord == glm::ivec3(numVoxels.x - 1, 0, numVoxels.z - 1) ||
              coord == glm::ivec3(numVoxels.x - 1, numVoxels.y - 1, 0) ||
              coord == glm::ivec3(numVoxels.x - 1, numVoxels.y - 1, numVoxels.z - 1)) {
            m_data->accessor.setValue(openvdb::Coord(x, y, z), reinterpret_cast<const openvdb::Vec4d&>(filteredData[id]));
          }
        }
      }
    }
  }

  void COpenvdbBackend::writeToFile(const char* directory, const char* fileName, const glm::uvec3& numVoxels) const {
    if (!m_data) {
      throw new std::exception("Called COpenvdbBackend::getNanoGrid without initializing COpenvdbBackend backend.");
    }

    auto handle = nanovdb::openToNanoVDB(m_data->grid);
    auto* dstGrid = handle.grid<nanovdb::Vec4d>();
    if (!dstGrid)
      throw std::runtime_error("GridHandle does not contain a grid with value type vec4d");

    std::shared_ptr<rt::CHostBrickGrid> gridBrick = std::make_shared<rt::CHostBrickGrid>(*m_data, numVoxels);

    std::filesystem::path baseDir(directory);
    std::filesystem::create_directories(baseDir);
    baseDir.append(fileName);

    std::filesystem::path nanoPath = baseDir;
    nanoPath.concat(".nvdb");
    nanovdb::io::writeGrid(nanoPath.string(), handle);

    std::filesystem::path brickGridPath = baseDir;
    brickGridPath.concat(".brickgrid");
    rt::write_grid(gridBrick, brickGridPath.string());
  }

  COpenvdbBackend* COpenvdbBackend::instance() {
    if (!s_instance) {
      s_instance = new COpenvdbBackend();
    }
    return s_instance;
  }

}
