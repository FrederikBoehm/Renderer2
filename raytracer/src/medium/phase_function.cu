#include "medium/phase_function.hpp"
#include "medium/henyey_greenstein_phase_function.hpp"
#include "medium/sggx_phase_function.hpp"
#include <stdio.h>
#include "sampling/sampler.hpp"

namespace rt {
  CPhaseFunction::CPhaseFunction(const EPhaseFunction type) :
    m_type(type) {

  }

  CPhaseFunction::~CPhaseFunction() {

  }
}