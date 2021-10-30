#include "integrators/direct_lighting_integrator.hpp"
#include "sampling/sampler.hpp"
#include "scene/scene.hpp"
#include "camera/pixel_sampler.hpp"
#include <cmath>
#include "sampling/mis.hpp"
#include "scene/interaction.hpp"
#include "shapes/circle.hpp"
#include "shapes/sphere.hpp"

namespace rt {
  CDirectLightingIntegrator::CDirectLightingIntegrator(CDeviceScene* scene, CPixelSampler* pixelSampler, CSampler* sampler, uint16_t numSamples):
    m_scene(scene),
    m_pixelSampler(pixelSampler),
    m_sampler(sampler),
    m_numSamples(numSamples) {

  }

  glm::vec3 CDirectLightingIntegrator::Li() {
    glm::vec3 L(0.0f);
    CRay eyeRay = m_pixelSampler->samplePixel();

    SInteraction si = m_scene->intersect(eyeRay);
    if (si.hitInformation.hit) {
      glm::vec3 trEye;
      if (si.medium) {
        si = m_scene->intersectTr(CRay(si.hitInformation.pos + 1e-6f * eyeRay.m_direction, eyeRay.m_direction), *m_sampler, &trEye);
      }
      else {
        trEye = glm::vec3(1.f);
      }
      //return glm::vec3(trEye) / glm::vec3(m_numSamples);
      if (si.material) {
        if (si.material->Le() != glm::vec3(0.0f)) { // Hit on light source
          L = si.material->Le() * trEye / (float)m_numSamples;
        }
        else {
          // Construct tangent space
          glm::vec3 notN = glm::normalize(glm::vec3(si.hitInformation.normal.x + 1.0f, si.hitInformation.normal.x + 2.0f, si.hitInformation.normal.x + 3.0f));
          glm::vec3 tangent = glm::normalize(glm::cross(notN, si.hitInformation.normal));
          glm::vec3 bitangent = glm::normalize(glm::cross(si.hitInformation.normal, tangent));

          glm::mat4 tangentToWorld(glm::vec4(tangent, 0.0f), glm::vec4(bitangent, 0.0f), glm::vec4(si.hitInformation.normal, 0.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
          glm::mat4 worldToTangent = glm::inverse(tangentToWorld);

          CRay eyeRayTangent = eyeRay.transform(worldToTangent);

          // Sample light source
          {
            float lightsPdf = 0.0f;
            glm::vec3 lightWorldSpaceDirection;
            float lightPdf;
            glm::vec3 Le = m_scene->sampleLightSources(*m_sampler, &lightWorldSpaceDirection, &lightPdf);
            glm::vec3 lightTangentSpaceDirection = glm::normalize(glm::vec3(worldToTangent * glm::vec4(lightWorldSpaceDirection, 0.0f)));

            CRay shadowRayLight = CRay(si.hitInformation.pos + 1.0e-6f * si.hitInformation.normal, lightWorldSpaceDirection);
            glm::vec3 trSecondary;
            SInteraction siLight = m_scene->intersectTr(shadowRayLight, *m_sampler, &trSecondary); // TODO: Handle case that second hit is on volume


            float brdfPdf = si.material->pdf(-eyeRayTangent.m_direction, lightTangentSpaceDirection);
            float mis_weight = balanceHeuristic(1, lightPdf, 1, brdfPdf);

            glm::vec3 f = si.material->f(-eyeRayTangent.m_direction, lightTangentSpaceDirection);
            glm::vec3 tr;
            if (!siLight.hitInformation.hit) {
              tr = trSecondary * trEye; // Light From environment map
            }
            else {
              tr = siLight.material && glm::vec3(0.f) != siLight.material->Le() ? glm::vec3(1.f) * trEye: glm::vec3(0.f);
            }
          
            if (lightPdf > 0.f) {
              L += mis_weight * f * Le * glm::max(glm::dot(si.hitInformation.normal, shadowRayLight.m_direction), 0.0f) * tr / ((float)m_numSamples * lightPdf);
              //L += tr;
            }
          }
            

          // Sample BRDF
          {
            glm::vec3 wi(0.0f);
            float brdfPdf;
            glm::vec3 f = si.material->sampleF(-eyeRayTangent.m_direction, &wi, *m_sampler, &brdfPdf);
            glm::vec3 brdfWorldSpaceDirection = glm::normalize(glm::vec3(tangentToWorld * glm::vec4(wi, 0.0f)));
            CRay shadowRayBrdf = CRay(si.hitInformation.pos + 1.0e-6f * si.hitInformation.normal, brdfWorldSpaceDirection);
            glm::vec3 trSecondary;
            SInteraction siBrdf = m_scene->intersectTr(shadowRayBrdf, *m_sampler, &trSecondary);

            if (siBrdf.hitInformation.hit) {
              if (siBrdf.material) {
                // TODO repair lighting from area lights
                //glm::vec3 Le = siBrdf.material->Le();
              
                //if (Le != glm::vec3(0.0f)) {
                //  float cosine = glm::max(glm::dot(si.hitInformation.normal, shadowRayBrdf.m_direction), 0.0f);
                //  float lightSamplingPdf = m_scene->lightSourcePdf(si2, shadowRayBrdf);
                //  lightsPdf = m_scene->lightSourcesPdf(si2);
                //  lightPdf = lightSamplingPdf * lightsPdf;
                //  if (lightPdf > 0.0f) {
                //    mis_weight = balanceHeuristic(1, brdfPdf, 1, lightPdf);

                //    L += mis_weight * f * Le * cosine * tr/ ((float)m_numSamples * brdfPdf);
                //  }
                //}
              }
            }
            else {
              float lightPdf;
              glm::vec3 Le = m_scene->le(brdfWorldSpaceDirection, &lightPdf);
              if (lightPdf > 0.0f) {
                float cosine = glm::max(glm::dot(si.hitInformation.normal, shadowRayBrdf.m_direction), 0.0f);
                float mis_weight = balanceHeuristic(1, brdfPdf, 1, lightPdf);

                glm::vec3 tr;
                if (!siBrdf.hitInformation.hit) {
                  tr = trSecondary * trEye; // Light From environment map
                }
                else {
                  tr = siBrdf.material && glm::vec3(0.f) != siBrdf.material->Le() ? glm::vec3(1.f) * trEye : glm::vec3(0.f);
                }

                L += mis_weight * f * Le * cosine * tr / ((float)m_numSamples * brdfPdf);
              }
            }
          }
          
        }
      }
      else {
        float p;
        L += m_scene->le(eyeRay.m_direction, &p) * trEye / (float)m_numSamples;
      }
    }
    else {
      float p;
      L += m_scene->le(eyeRay.m_direction, &p) / (float)m_numSamples;
    }
    return L;
  }
}