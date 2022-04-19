#include "medium/medium_instance.hpp"
#include "medium/nvdb_medium.hpp"

namespace rt {
  CMediumInstance::CMediumInstance(CNVDBMedium* medium, const glm::mat4x3* modelToWorld, const glm::mat4x3* worldToModel) :
    m_medium(medium),
    m_modelToWorld(modelToWorld),
    m_worldToModel(worldToModel) {
  }

  std::string CMediumInstance::path() const {
    return m_medium->path();
  }

  OptixTraversableHandle CMediumInstance::getOptixHandle() const {
    return m_medium->getOptixHandle();
  }

  OptixProgramGroup CMediumInstance::getOptixProgramGroup() const {
    return m_medium->getOptixProgramGroup();
  }

  void CMediumInstance::setFilterRenderRatio(float ratio) {
    m_filterRenderRatio = ratio;
  }

  float CMediumInstance::filterRenderRatio() const {
    return m_filterRenderRatio;
  }
}