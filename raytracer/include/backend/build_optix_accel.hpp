#ifndef BUILD_OPTIX_ACCEL_HPP
#define BUILD_OPTIX_ACCEL_HPP
#include <optix_types.h>
namespace rt {
  void buildOptixAccel(const OptixBuildInput& buildInput, OptixTraversableHandle* traversableHandle, CUdeviceptr* deviceGasBuffer);

}
#endif