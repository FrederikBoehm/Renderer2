#define _USE_MATH_DEFINES
#include <math.h>

#include "medium/sggx_phase_function.hpp"
#include "integrators/objects.hpp"
#include "sampling/sampler.hpp"
#include <stdio.h>

namespace rt {
  // CSGGXMicroflakeDistribution

  CSGGXMicroflakeDistribution::CSGGXMicroflakeDistribution(const SSGGXDistributionParameters& parameters) :
    m_S(getS(parameters)),
    m_invS(glm::inverse(m_S)),
    m_sigma(0.f) {

  }

  CSGGXMicroflakeDistribution::CSGGXMicroflakeDistribution(float sigma) :
    m_S(1.f),
    m_invS(1.f),
    m_sigma(sigma) {

  }

  // CSGGXSpecularPhaseFunction

  CSGGXSpecularPhaseFunction::CSGGXSpecularPhaseFunction(const SSGGXDistributionParameters& distributionsParameters) :
    m_distribution(distributionsParameters) {

  }

  CSGGXSpecularPhaseFunction::CSGGXSpecularPhaseFunction(float sigma) :
    m_distribution(sigma) {

  }

  // CSGGXDiffusePhaseFunction

  CSGGXDiffusePhaseFunction::CSGGXDiffusePhaseFunction(const SSGGXDistributionParameters& distributionParameters) :
    m_distribution(distributionParameters) {

  }

  CSGGXDiffusePhaseFunction::CSGGXDiffusePhaseFunction(float sigma) :
    m_distribution(sigma) {

  }

  // CSGGXPhaseFunction

  CSGGXPhaseFunction::CSGGXPhaseFunction(const SSGGXDistributionParameters& diffuseParameters, const SSGGXDistributionParameters& specularParameters) :
    CPhaseFunction(EPhaseFunction::SGGX),
    m_diffuse(diffuseParameters),
    m_specular(specularParameters) {

  }

  CSGGXPhaseFunction::CSGGXPhaseFunction(float diffuseSigma, float specularSigma) :
    CPhaseFunction(EPhaseFunction::SGGX),
    m_diffuse(diffuseSigma),
    m_specular(specularSigma) {

  }
}