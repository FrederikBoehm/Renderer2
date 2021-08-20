#include "integrators/direct_lighting_integrator.hpp"
#include "sampling/sampler.hpp"
#include "scene/scene.hpp"
#include "camera/pixel_sampler.hpp"

namespace rt {
  CDirectLightingIntegrator::CDirectLightingIntegrator(CDeviceScene* scene, CPixelSampler* pixelSampler, CSampler* sampler, uint16_t numSamples):
    m_scene(scene),
    m_pixelSampler(pixelSampler),
    m_sampler(sampler),
    m_numSamples(numSamples) {

  }

  glm::vec3 CDirectLightingIntegrator::Li(EIntegrationStrategy strategy) {
    glm::vec3 L(0.0f);
    if (strategy == UNIFORM_SAMPLE_HEMISPHERE) {
      Ray eyeRay = m_pixelSampler->samplePixel();

      SSurfaceInteraction si = m_scene->intersect(eyeRay);
      if (si.hitInformation.hit) {
        if (si.material.Le() != glm::vec3(0.0f)) { // Hit on light source
          L = si.material.Le() / (float)m_numSamples;
        }
        else {
          glm::vec3 tangentSpaceDirection = m_sampler->uniformSampleHemisphere();
          // Construct tangent space
          glm::vec3 notN = glm::normalize(glm::vec3(si.hitInformation.normal.x + 1.0f, si.hitInformation.normal.x + 2.0f, si.hitInformation.normal.x + 3.0f));
          glm::vec3 tangent = glm::normalize(glm::cross(notN, si.hitInformation.normal));
          glm::vec3 bitangent = glm::normalize(glm::cross(si.hitInformation.normal, tangent));

          glm::mat4 tangentToWorld(glm::vec4(tangent, 0.0f), glm::vec4(si.hitInformation.normal, 0.0f), glm::vec4(bitangent, 0.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
          glm::mat4 worldToTangent = glm::inverse(tangentToWorld);
          glm::vec3 worldSpaceDirection = glm::normalize(glm::vec3(tangentToWorld * glm::vec4(tangentSpaceDirection, 0.0f)));

          Ray shadowRay = Ray(si.hitInformation.pos + FLT_EPSILON * si.hitInformation.normal, worldSpaceDirection);
          SSurfaceInteraction si2 = m_scene->intersect(shadowRay);

          Ray eyeRayTangent = eyeRay.transform(worldToTangent);

          glm::vec3 f = si.material.f(si.hitInformation, -eyeRayTangent.m_direction, tangentSpaceDirection);
          glm::vec3 Le = si2.material.Le();
          float cosine = glm::max(glm::dot(si.hitInformation.normal, shadowRay.m_direction), 0.0f);
          float pdf = m_sampler->uniformHemispherePdf();

          L = f * Le * cosine / ((float)m_numSamples * pdf);
        }
      }
    }
    else {
    }
    return L;
  }
}