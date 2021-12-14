#define _USE_MATH_DEFINES
#include <math.h>
#include "medium/henyey_greenstein_phase_function.hpp"
#include <algorithm>
#include "utility/functions.hpp"
#include <glm/glm.hpp>

namespace rt {

  CHenyeyGreensteinPhaseFunction::CHenyeyGreensteinPhaseFunction(float g) :
    CPhaseFunction(EPhaseFunction::HENYEY_GREENSTEIN),
    m_g(g) {

  }
}

