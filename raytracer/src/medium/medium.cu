#include "medium/medium.hpp"

#include "medium/homogeneous_medium.hpp"
#include <stdio.h>
#include "intersect/ray.hpp"
#include "sampling/sampler.hpp"
#include "scene/interaction.hpp"
#include "medium/heterogenous_medium.hpp"
#include "medium/nvdb_medium.hpp"
#include "medium/phase_function.hpp"

namespace rt {
  CMedium::CMedium(const EMediumType type) :
    m_type(type) {

  }

  CMedium::~CMedium() {

  }

  OptixProgramGroup CMedium::getOptixProgramGroup() const {
    switch (m_type) {
    case NVDB_MEDIUM:
      return static_cast<const CNVDBMedium*>(this)->getOptixProgramGroup();
    }
    fprintf(stderr, "[ERROR] No OptixProgramGroup found for given medium type\n");
    return OptixProgramGroup();
  }

  std::string CMedium::path() const {
    switch (m_type) {
    case NVDB_MEDIUM:
      return static_cast<const CNVDBMedium*>(this)->path();
    }
    printf("[ERROR] No path found for given medium type\n");
    return "";
  }

  OptixTraversableHandle CMedium::getOptixHandle() const {
    switch (m_type) {
    case NVDB_MEDIUM:
      return static_cast<const CNVDBMedium*>(this)->getOptixHandle();
    }
    printf("[ERROR] No OptixTraversableHandle found for given medium type\n");
    return NULL;
  }
}