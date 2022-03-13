#define _USE_MATH_DEFINES
#include <math.h>

#include "medium/sggx_phase_function.hpp"
#include "integrators/objects.hpp"
#include "sampling/sampler.hpp"

namespace rt {
  CSGGXPhaseFunctionPlaceholder::CSGGXPhaseFunctionPlaceholder() :
    CPhaseFunction(EPhaseFunction::SGGX) {

  }
}