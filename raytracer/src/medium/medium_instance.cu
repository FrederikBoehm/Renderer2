#include "medium/medium_instance.hpp"
#include "medium/nvdb_medium.hpp"

namespace rt {
  CMediumInstance::CMediumInstance(CNVDBMedium* medium, const glm::mat4* modelToWorld, const glm::mat4* worldToModel) :
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
}