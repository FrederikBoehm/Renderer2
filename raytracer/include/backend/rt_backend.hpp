#ifndef RT_BACKEND_HPP
#define RT_BACKEND_HPP
#include <optix/optix_types.h>
#include "utility/qualifiers.hpp"
#include <vector>
#include "types.hpp"

#define RAY_TYPE_COUNT 1

namespace rt {
  // Uses large parts of optixVolumeViewer sample

  class CDeviceSceneobject;

  class CRTBackend {
  public:
    struct SProgramGroups {
      OptixProgramGroup m_raygen;
      OptixProgramGroup m_miss;
      OptixProgramGroup m_hitSurface;
      OptixProgramGroup m_hitVolume;
      OptixProgramGroup m_hitMesh;
    };

    struct SProgramGroupFunctionNames {
      const char* raygen;
      const char* miss;
      const char* closesthit;
      const char* intersectSurface;
      const char* intersectionVolume;
      const char* anyhitMesh;
    };

    H_CALLABLE void init();
    H_CALLABLE void release();
    H_CALLABLE void reset();
    H_CALLABLE void createModule(const std::string& ptxFile, const char* launchParams = "params");
    H_CALLABLE static CRTBackend* instance();
    H_CALLABLE const OptixDeviceContext& context();
    H_CALLABLE void createProgramGroups(const SProgramGroupFunctionNames& functionNames = { "__raygen__rg", "__miss__ms", "__closesthit__ch", "__intersection__surface", "__intersection__volume", "__anyhit__mesh" });
    H_CALLABLE void createPipeline(const char* launchParams = "params");
    H_CALLABLE void createSBT(const std::vector<SRecord<const CDeviceSceneobject*>>& hitgroupRecords);

    H_CALLABLE OptixPipeline& pipeline();
    H_CALLABLE OptixShaderBindingTable& sbt();
    H_CALLABLE const SProgramGroups& programGroups() const;

  private:
    static CRTBackend* s_instance;

    OptixDeviceContext m_context;
    OptixModule m_module;
    SProgramGroups m_programGroups;
    OptixPipeline m_pipeline;
    OptixShaderBindingTable m_sbt;
    
    H_CALLABLE CRTBackend() = default;
    H_CALLABLE CRTBackend(const CRTBackend&) = delete;
    H_CALLABLE CRTBackend(CRTBackend&&) = delete;
    H_CALLABLE CRTBackend& operator=(const CRTBackend&) = delete;
    H_CALLABLE CRTBackend& operator=(CRTBackend&&) = delete;
    H_CALLABLE OptixPipelineCompileOptions getCompileOptions(const char* launchParams);
  };

  inline const OptixDeviceContext& CRTBackend::context() {
    return m_context;
  }

  inline OptixPipeline& CRTBackend::pipeline() {
    return m_pipeline;
  }

  inline OptixShaderBindingTable& CRTBackend::sbt() {
    return m_sbt;
  }

  inline const CRTBackend::SProgramGroups& CRTBackend::programGroups() const {
    return m_programGroups;
  }
}
#endif // !RT_BACKEND_HPP
