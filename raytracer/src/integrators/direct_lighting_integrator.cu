#include "integrators/direct_lighting_integrator.hpp"
#include "sampling/sampler.hpp"
#include "scene/scene.hpp"
#include "camera/pixel_sampler.hpp"
#include <cmath>
#include "sampling/mis.hpp"
#include "scene/interaction.hpp"
#include "shapes/plane.hpp"
#include "shapes/sphere.hpp"

namespace rt {
  CDirectLightingIntegrator::CDirectLightingIntegrator(CDeviceScene* scene, CPixelSampler* pixelSampler, CSampler* sampler, uint16_t numSamples):
    m_scene(scene),
    m_pixelSampler(pixelSampler),
    m_sampler(sampler),
    m_numSamples(numSamples) {

  }

  glm::vec3 CDirectLightingIntegrator::Li(EIntegrationStrategy strategy) {
    glm::vec3 L(0.0f);
    //if (strategy == UNIFORM_SAMPLE_HEMISPHERE) {
    //  CRay eyeRay = m_pixelSampler->samplePixel();

    //  SInteraction si = m_scene->intersect(eyeRay);
    //  if (si.hitInformation.hit) {
    //    if (si.material) { // Hit on geometry
    //      if (si.material->Le() != glm::vec3(0.0f)) { // Hit on light source
    //        L = si.material->Le() / (float)m_numSamples;
    //      }
    //      else {
    //        glm::vec3 tangentSpaceDirection = m_sampler->uniformSampleHemisphere();
    //        // Construct tangent space
    //        glm::vec3 notN = glm::normalize(glm::vec3(si.hitInformation.normal.x + 1.0f, si.hitInformation.normal.x + 2.0f, si.hitInformation.normal.x + 3.0f));
    //        glm::vec3 tangent = glm::normalize(glm::cross(notN, si.hitInformation.normal));
    //        glm::vec3 bitangent = glm::normalize(glm::cross(si.hitInformation.normal, tangent));

    //        glm::mat4 tangentToWorld(glm::vec4(tangent, 0.0f), glm::vec4(bitangent, 0.0f), glm::vec4(si.hitInformation.normal, 0.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    //        glm::mat4 worldToTangent = glm::inverse(tangentToWorld);
    //        glm::vec3 worldSpaceDirection = glm::normalize(glm::vec3(tangentToWorld * glm::vec4(tangentSpaceDirection, 0.0f)));

    //        CRay shadowRay = CRay(si.hitInformation.pos + 1.0e-6f * si.hitInformation.normal, worldSpaceDirection);
    //        SInteraction si2 = m_scene->intersect(shadowRay); // TODO: Handle case that second hit is on volume

    //        CRay eyeRayTangent = eyeRay.transform(worldToTangent);

    //        glm::vec3 f = si.material->f(-eyeRayTangent.m_direction, tangentSpaceDirection);
    //        glm::vec3 Le = si2.material->Le();
    //        float cosine = glm::max(glm::dot(si.hitInformation.normal, shadowRay.m_direction), 0.0f);
    //        float pdf = m_sampler->uniformHemispherePdf();

    //        L = f * Le * cosine / ((float)m_numSamples * pdf);
    //      }
    //    }
    //  }
    //}
    //else if (strategy == IMPORTANCE_SAMPLE_LIGHTSOURCES) {
    //  CRay eyeRay = m_pixelSampler->samplePixel();

    //  SInteraction si = m_scene->intersect(eyeRay);
    //  if (si.hitInformation.hit) {
    //    if (si.material) {
    //      if (si.material->Le() != glm::vec3(0.0f)) { // Hit on light source
    //        L = si.material->Le() / (float)m_numSamples;
    //      }
    //      else {
    //        // Construct tangent space
    //        glm::vec3 notN = glm::normalize(glm::vec3(si.hitInformation.normal.x + 1.0f, si.hitInformation.normal.x + 2.0f, si.hitInformation.normal.x + 3.0f));
    //        glm::vec3 tangent = glm::normalize(glm::cross(notN, si.hitInformation.normal));
    //        glm::vec3 bitangent = glm::normalize(glm::cross(si.hitInformation.normal, tangent));

    //        glm::mat4 tangentToWorld(glm::vec4(tangent, 0.0f), glm::vec4(bitangent, 0.0f), glm::vec4(si.hitInformation.normal, 0.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    //        glm::mat4 worldToTangent = glm::inverse(tangentToWorld);

    //        float lightsPdf;
    //        glm::vec3 pos = m_scene->sampleLightSources(*m_sampler, &lightsPdf);
    //        glm::vec3 worldSpaceDirection = glm::normalize(m_scene->sampleLightSources(*m_sampler, &lightsPdf) - si.hitInformation.pos);
    //        glm::vec3 tangentSpaceDirection = glm::normalize(glm::vec3(worldToTangent * glm::vec4(worldSpaceDirection, 0.0f)));

    //        CRay shadowRay = CRay(si.hitInformation.pos + 1.0e-6f * si.hitInformation.normal, worldSpaceDirection);
    //        SInteraction si2 = m_scene->intersect(shadowRay); // TODO: Handle case that second hit is on volume

    //        CRay eyeRayTangent = eyeRay.transform(worldToTangent);

    //        float lightSamplingPdf = m_scene->lightSourcePdf(si2, shadowRay);
    //        glm::vec3 f = si.material->f(-eyeRayTangent.m_direction, tangentSpaceDirection);
    //        glm::vec3 Le = si2.material->Le();
    //        float distance = glm::length(si.hitInformation.pos - si2.hitInformation.pos);
    //        float G = glm::max(glm::dot(si.hitInformation.normal, shadowRay.m_direction), 0.0f) * glm::max(glm::dot(si2.hitInformation.normal, -shadowRay.m_direction), 0.0f) / (distance * distance);

    //        L = f * Le * G / ((float)m_numSamples * lightsPdf * lightSamplingPdf);
    //      }
    //    }
    //  }
    //}
    //else if (strategy == IMPORTANCE_SAMPLE_COSINE) {
    //  CRay eyeRay = m_pixelSampler->samplePixel();

    //  SInteraction si = m_scene->intersect(eyeRay);
    //  if (si.hitInformation.hit) {
    //    if (si.material->Le() != glm::vec3(0.0f)) { // Hit on light source
    //      L = si.material->Le() / (float)m_numSamples;
    //    }
    //    else {
    //      glm::vec3 tangentSpaceDirection = m_sampler->cosineSampleHemisphere();
    //      // Construct tangent space
    //      glm::vec3 notN = glm::normalize(glm::vec3(si.hitInformation.normal.x + 1.0f, si.hitInformation.normal.x + 2.0f, si.hitInformation.normal.x + 3.0f));
    //      glm::vec3 tangent = glm::normalize(glm::cross(notN, si.hitInformation.normal));
    //      glm::vec3 bitangent = glm::normalize(glm::cross(si.hitInformation.normal, tangent));

    //      glm::mat4 tangentToWorld(glm::vec4(tangent, 0.0f), glm::vec4(bitangent, 0.0f), glm::vec4(si.hitInformation.normal, 0.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    //      glm::mat4 worldToTangent = glm::inverse(tangentToWorld);
    //      glm::vec3 worldSpaceDirection = glm::normalize(glm::vec3(tangentToWorld * glm::vec4(tangentSpaceDirection, 0.0f)));

    //      CRay shadowRay = CRay(si.hitInformation.pos + 1.0e-6f * si.hitInformation.normal, worldSpaceDirection);
    //      SInteraction si2 = m_scene->intersect(shadowRay); // TODO: Handle case that second hit is on volume

    //      CRay eyeRayTangent = eyeRay.transform(worldToTangent);

    //      glm::vec3 f = si.material->f(-eyeRayTangent.m_direction, tangentSpaceDirection);
    //      glm::vec3 Le = si2.material->Le();
    //      float cosine = glm::max(glm::dot(si.hitInformation.normal, shadowRay.m_direction), 0.0f);
    //      float pdf = m_sampler->cosineHemispherePdf(cosine);

    //      L = f * Le * cosine / ((float)m_numSamples * pdf);
    //    }
    //  }
    //}
    //else if (strategy == IMPORTANCE_SAMPLE_BRDF) {
    //  CRay eyeRay = m_pixelSampler->samplePixel();

    //  SInteraction si = m_scene->intersect(eyeRay);
    //  if (si.hitInformation.hit) {
    //    if (si.material) {
    //      if (si.material->Le() != glm::vec3(0.0f)) { // Hit on light source
    //        L = si.material->Le() / (float)m_numSamples;
    //      }
    //      else {
    //        // Construct tangent space
    //        glm::vec3 notN = glm::normalize(glm::vec3(si.hitInformation.normal.x + 1.0f, si.hitInformation.normal.x + 2.0f, si.hitInformation.normal.x + 3.0f));
    //        glm::vec3 tangent = glm::normalize(glm::cross(notN, si.hitInformation.normal));
    //        glm::vec3 bitangent = glm::normalize(glm::cross(si.hitInformation.normal, tangent));

    //        glm::mat4 tangentToWorld(glm::vec4(tangent, 0.0f), glm::vec4(bitangent, 0.0f), glm::vec4(si.hitInformation.normal, 0.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    //        glm::mat4 worldToTangent = glm::inverse(tangentToWorld);


    //        CRay eyeRayTangent = eyeRay.transform(worldToTangent);
    //        glm::vec3 wi;
    //        float pdf;
    //        //glm::vec3 f = si.material.sampleF(-eyeRayTangent.m_direction, &wi, *m_sampler, &pdf);
    //        si.material->sampleF(-eyeRayTangent.m_direction, &wi, *m_sampler, &pdf);
    //        glm::vec3 f = si.material->f(-eyeRayTangent.m_direction, wi);
    //      
    //        glm::vec3 worldSpaceDirection = glm::normalize(glm::vec3(tangentToWorld * glm::vec4(wi, 0.0f)));
    //        CRay shadowRay = CRay(si.hitInformation.pos + 1.0e-6f * si.hitInformation.normal, worldSpaceDirection);
    //        SInteraction si2 = m_scene->intersect(shadowRay); // TODO: Handle case that second hit is on volume

    //        glm::vec3 Le = si2.material->Le();
    //        float cosine = glm::max(glm::dot(si.hitInformation.normal, shadowRay.m_direction), 0.0f);

    //        L = f * Le * cosine / ((float)m_numSamples * pdf);
    //      }
    //    }
    //  }
    //}
    //else if (strategy == MULTIPLE_IMPORTANCE_SAMPLE) {
      CRay eyeRay = m_pixelSampler->samplePixel();

      glm::vec3 trEye;
      SInteraction si = m_scene->intersect(eyeRay);
      if (si.hitInformation.hit) {
        if (si.material) {
          if (si.material->Le() != glm::vec3(0.0f)) { // Hit on light source
            L = si.material->Le() / (float)m_numSamples;
          }
          else {
            // Construct tangent space
            glm::vec3 notN = glm::normalize(glm::vec3(si.hitInformation.normal.x + 1.0f, si.hitInformation.normal.x + 2.0f, si.hitInformation.normal.x + 3.0f));
            glm::vec3 tangent = glm::normalize(glm::cross(notN, si.hitInformation.normal));
            glm::vec3 bitangent = glm::normalize(glm::cross(si.hitInformation.normal, tangent));

            glm::mat4 tangentToWorld(glm::vec4(tangent, 0.0f), glm::vec4(bitangent, 0.0f), glm::vec4(si.hitInformation.normal, 0.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
            glm::mat4 worldToTangent = glm::inverse(tangentToWorld);

            // Sample light source
            float lightsPdf = 0.0f;
            //glm::vec3 lightPos = m_scene->sampleLightSources(*m_sampler, &lightsPdf);
            //glm::vec3 lightWorldSpaceDirection = glm::normalize(m_scene->sampleLightSources(*m_sampler, &lightsPdf) - si.hitInformation.pos);
            glm::vec3 lightWorldSpaceDirection;
            float lightPdf;
            glm::vec3 Le = m_scene->sampleLightSources(*m_sampler, &lightWorldSpaceDirection, &lightPdf);
            glm::vec3 lightTangentSpaceDirection = glm::normalize(glm::vec3(worldToTangent * glm::vec4(lightWorldSpaceDirection, 0.0f)));

            CRay shadowRayLight = CRay(si.hitInformation.pos + 1.0e-6f * si.hitInformation.normal, lightWorldSpaceDirection);
            SInteraction si2 = m_scene->intersect(shadowRayLight); // TODO: Handle case that second hit is on volume

            CRay eyeRayTangent = eyeRay.transform(worldToTangent);

            //float lightSamplingPdf = m_scene->lightSourcePdf(si2, shadowRayLight);
            //float lightPdf = lightsPdf * lightSamplingPdf;
            float brdfPdf = si.material->pdf(-eyeRayTangent.m_direction, lightTangentSpaceDirection);
            float mis_weight = balanceHeuristic(1, lightPdf, 1, brdfPdf);

            //glm::vec3 f = si.material->f(-eyeRayTangent.m_direction, lightTangentSpaceDirection);
            //if (si2.hitInformation.hit && si2.material && lightPdf > 0.0f) {
            //  glm::vec3 Le = si2.material->Le();
            //  float distance = glm::length(si.hitInformation.pos - si2.hitInformation.pos);
            //  float G = glm::max(glm::dot(si.hitInformation.normal, shadowRayLight.m_direction), 0.0f) * glm::max(glm::dot(si2.hitInformation.normal, -shadowRayLight.m_direction), 0.0f) / (distance * distance);

            //  L += mis_weight * f * Le * G / ((float)m_numSamples * lightPdf);
            //}
            glm::vec3 f = si.material->f(-eyeRayTangent.m_direction, lightTangentSpaceDirection);
            float V = !si2.hitInformation.hit;
            if (lightPdf > 0.f) {
              L += mis_weight * f * Le * glm::max(glm::dot(si.hitInformation.normal, shadowRayLight.m_direction), 0.0f) * V / ((float)m_numSamples * lightPdf);
              //L += Le;
            }

            //if (si2.hitInformation.hit && si2.material && lightPdf > 0.0f) {
            //  glm::vec3 Le = si2.material->Le();
            //  float distance = glm::length(si.hitInformation.pos - si2.hitInformation.pos);
            //  float G = glm::max(glm::dot(si.hitInformation.normal, shadowRayLight.m_direction), 0.0f) * glm::max(glm::dot(si2.hitInformation.normal, -shadowRayLight.m_direction), 0.0f) / (distance * distance);

            //  L += mis_weight * f * Le * G / ((float)m_numSamples * lightPdf);
            //}
            

            // Sample BRDF
            glm::vec3 wi(0.0f);
            f = si.material->sampleF(-eyeRayTangent.m_direction, &wi, *m_sampler, &brdfPdf);
            glm::vec3 brdfWorldSpaceDirection = glm::normalize(glm::vec3(tangentToWorld * glm::vec4(wi, 0.0f)));
            CRay shadowRayBrdf = CRay(si.hitInformation.pos + 1.0e-6f * si.hitInformation.normal, brdfWorldSpaceDirection);
            si2 = m_scene->intersect(shadowRayBrdf);

            if (si2.hitInformation.hit) {
              //if (si2.material) {
              //  glm::vec3 Le = si2.material->Le();
              //
              //  if (Le != glm::vec3(0.0f)) {
              //    float cosine = glm::max(glm::dot(si.hitInformation.normal, shadowRayBrdf.m_direction), 0.0f);
              //    lightSamplingPdf = m_scene->lightSourcePdf(si2, shadowRayBrdf);
              //    lightsPdf = m_scene->lightSourcesPdf(si2);
              //    lightPdf = lightSamplingPdf * lightsPdf;
              //    if (lightPdf > 0.0f) {
              //      mis_weight = balanceHeuristic(1, brdfPdf, 1, lightPdf);

              //      L += mis_weight * f * Le * cosine / ((float)m_numSamples * brdfPdf);
              //    }
              //  }
              //}
            }
            else {
              float pdf;
              glm::vec3 Le = m_scene->le(brdfWorldSpaceDirection, &pdf);
              float cosine = glm::max(glm::dot(si.hitInformation.normal, shadowRayBrdf.m_direction), 0.0f);
              mis_weight = balanceHeuristic(1, brdfPdf, 1, lightPdf);
              if (lightPdf > 0.0f) {
                L += mis_weight * f * Le * cosine / ((float)m_numSamples * brdfPdf);
              }
            }
          
          }
        }
        else if (si.medium) {
          // Construct tangent space
          glm::vec3 notN = glm::normalize(glm::vec3(si.hitInformation.normal.x + 1.0f, si.hitInformation.normal.x + 2.0f, si.hitInformation.normal.x + 3.0f));
          glm::vec3 tangent = glm::normalize(glm::cross(notN, si.hitInformation.normal));
          glm::vec3 bitangent = glm::normalize(glm::cross(si.hitInformation.normal, tangent));

          glm::mat4 tangentToWorld(glm::vec4(tangent, 0.0f), glm::vec4(bitangent, 0.0f), glm::vec4(si.hitInformation.normal, 0.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
          glm::mat4 worldToTangent = glm::inverse(tangentToWorld);

           // Sample light source
          {

            float lightsPdf = 0.0f;
            //glm::vec3 lightPos = m_scene->sampleLightSources(*m_sampler, &lightsPdf);
            //glm::vec3 lightWorldSpaceDirection = glm::normalize(m_scene->sampleLightSources(*m_sampler, &lightsPdf) - si.hitInformation.pos);
            glm::vec3 lightWorldSpaceDirection;
            float lightPdf;
            glm::vec3 le = m_scene->sampleLightSources(*m_sampler, &lightWorldSpaceDirection, &lightPdf);
            //glm::vec3 lightTangentSpaceDirection = glm::normalize(glm::vec3(worldToTangent * glm::vec4(lightWorldSpaceDirection, 0.0f)));

            //CRay shadowRay = CRay(si.hitInformation.pos + 1.0e-6f * si.hitInformation.normal, lightWorldSpaceDirection);
            CRay shadowRay = CRay(si.hitInformation.pos, lightWorldSpaceDirection);
            //SInteraction si2 = m_scene->intersect(shadowRay); // TODO: Handle case that second hit is on volume
            glm::vec3 tr;
            SInteraction si2 = m_scene->intersectTr(shadowRay, *m_sampler, &tr);
            //SInteraction si2 = m_scene->intersect(shadowRay);

            //CRay eyeRayTangent = eyeRay.transform(worldToTangent);

            float lightSamplingPdf = m_scene->lightSourcePdf(si2, shadowRay);
            //float lightPdf = lightsPdf * lightSamplingPdf;
            //float brdfPdf = si.material->pdf(-eyeRayTangent.m_direction, lightTangentSpaceDirection);
          
            float p = si.medium->phase().p(-eyeRay.m_direction, shadowRay.m_direction);
            float scatteringPdf = p;
            glm::vec3 f(p);
            //if (si2.material) {
            //  L += tr;
            //}
            if (lightPdf > 0.f) {
              float V = !si2.hitInformation.hit;

              float mis_weight = balanceHeuristic(1, lightPdf, 1, scatteringPdf);

              //glm::vec3 tr = m_scene->tr(shadowRayLight, *m_sampler);
              //glm::vec3 tr;
              //si2 = m_scene->intersectTr(shadowRay, *m_sampler, &tr);

              L += f * le * tr * mis_weight * V / lightPdf;
              //L += tr.r > tr.g ? glm::vec3(1.f, 0.f, 0.0f) : glm::vec3(0.f);
            }
          }

           // Sample phase function
          {
            glm::vec3 wi;
            float p = si.medium->phase().sampleP(-eyeRay.m_direction, &wi, glm::vec2(m_sampler->uniformSample01(), m_sampler->uniformSample01()));
            glm::vec3 f(p);
            float scatteringPdf = p;

            //CRay shadowRay = CRay(si.hitInformation.pos + 1.0e-6f * si.hitInformation.normal, wi);
            CRay shadowRay = CRay(si.hitInformation.pos, wi);
            //SInteraction si2 = m_scene->intersect(shadowRay);
            SInteraction si2;

              
            glm::vec3 tr;
            si2 = m_scene->intersectTr(shadowRay, *m_sampler, &tr);
            glm::vec3 le(0.f);
            float lightPdf = 0.f;
            if (si2.hitInformation.hit) {
              if (si2.material) {
                le = si2.material->Le();
                //lightPdf = m_scene->lightSourcesPdf(si2) * m_scene->lightSourcePdf(si2, shadowRay);
              }
            }
            else {
              le = m_scene->le(shadowRay.m_direction, &lightPdf);
            }

            if (lightPdf == 0.0f || scatteringPdf == 0.0f) {
              return L;
            }

            float mis_weight = balanceHeuristic(1, scatteringPdf, 1, lightPdf);
            L += f * le * tr * mis_weight / scatteringPdf;
            
            //float lightPdf = m_scene->lightSourcesPdf(si2) * m_scene->lightSourcePdf(si2, shadowRay);
            //float lightPdf;
            //glm::vec3 le = m_scene->le(shadowRay.m_direction, &lightPdf);

            //if (lightPdf == 0.0f || scatteringPdf == 0.0f) {
            //  return L;
            //}
            //
            //float mis_weight = balanceHeuristic(1, scatteringPdf, 1, lightPdf);

            //float V = !si2.hitInformation.hit;


            //L += f * le * tr * mis_weight / scatteringPdf;
            //L += tr.r > tr.g ? glm::vec3(1.f, 0.f, 0.0f) : glm::vec3(0.f);



          }
        }
      }
      else {
        float p;
        L += m_scene->le(eyeRay.m_direction, &p);
      }
    //}
    return L;
  }
}