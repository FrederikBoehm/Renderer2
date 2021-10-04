#include "integrators/direct_lighting_integrator.hpp"
#include "sampling/sampler.hpp"
#include "scene/scene.hpp"
#include "camera/pixel_sampler.hpp"
#include <cmath>
#include "sampling/mis.hpp"

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

          glm::mat4 tangentToWorld(glm::vec4(tangent, 0.0f), glm::vec4(bitangent, 0.0f), glm::vec4(si.hitInformation.normal, 0.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
          glm::mat4 worldToTangent = glm::inverse(tangentToWorld);
          glm::vec3 worldSpaceDirection = glm::normalize(glm::vec3(tangentToWorld * glm::vec4(tangentSpaceDirection, 0.0f)));

          Ray shadowRay = Ray(si.hitInformation.pos + 1.0e-6f * si.hitInformation.normal, worldSpaceDirection);
          SSurfaceInteraction si2 = m_scene->intersect(shadowRay);

          Ray eyeRayTangent = eyeRay.transform(worldToTangent);

          glm::vec3 f = si.material.f(-eyeRayTangent.m_direction, tangentSpaceDirection);
          glm::vec3 Le = si2.material.Le();
          float cosine = glm::max(glm::dot(si.hitInformation.normal, shadowRay.m_direction), 0.0f);
          float pdf = m_sampler->uniformHemispherePdf();

          L = f * Le * cosine / ((float)m_numSamples * pdf);
        }
      }
    }
    else if (strategy == IMPORTANCE_SAMPLE_LIGHTSOURCES) {
      Ray eyeRay = m_pixelSampler->samplePixel();

      SSurfaceInteraction si = m_scene->intersect(eyeRay);
      if (si.hitInformation.hit) {
        if (si.material.Le() != glm::vec3(0.0f)) { // Hit on light source
          L = si.material.Le() / (float)m_numSamples;
        }
        else {
          // Construct tangent space
          glm::vec3 notN = glm::normalize(glm::vec3(si.hitInformation.normal.x + 1.0f, si.hitInformation.normal.x + 2.0f, si.hitInformation.normal.x + 3.0f));
          glm::vec3 tangent = glm::normalize(glm::cross(notN, si.hitInformation.normal));
          glm::vec3 bitangent = glm::normalize(glm::cross(si.hitInformation.normal, tangent));

          glm::mat4 tangentToWorld(glm::vec4(tangent, 0.0f), glm::vec4(bitangent, 0.0f), glm::vec4(si.hitInformation.normal, 0.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
          glm::mat4 worldToTangent = glm::inverse(tangentToWorld);

          float lightsPdf;
          glm::vec3 pos = m_scene->sampleLightSources(*m_sampler, &lightsPdf);
          glm::vec3 worldSpaceDirection = glm::normalize(m_scene->sampleLightSources(*m_sampler, &lightsPdf) - si.hitInformation.pos);
          glm::vec3 tangentSpaceDirection = glm::normalize(glm::vec3(worldToTangent * glm::vec4(worldSpaceDirection, 0.0f)));

          Ray shadowRay = Ray(si.hitInformation.pos + 1.0e-6f * si.hitInformation.normal, worldSpaceDirection);
          SSurfaceInteraction si2 = m_scene->intersect(shadowRay);

          Ray eyeRayTangent = eyeRay.transform(worldToTangent);

          float lightSamplingPdf = m_scene->lightSourcePdf(si2, shadowRay);
          glm::vec3 f = si.material.f(-eyeRayTangent.m_direction, tangentSpaceDirection);
          glm::vec3 Le = si2.material.Le();
          float distance = glm::length(si.hitInformation.pos - si2.hitInformation.pos);
          float G = glm::max(glm::dot(si.hitInformation.normal, shadowRay.m_direction), 0.0f) * glm::max(glm::dot(si2.hitInformation.normal, -shadowRay.m_direction), 0.0f) / (distance * distance);

          L = f * Le * G / ((float)m_numSamples * lightsPdf * lightSamplingPdf);
        }
      }
    }
    else if (strategy == IMPORTANCE_SAMPLE_COSINE) {
      Ray eyeRay = m_pixelSampler->samplePixel();

      SSurfaceInteraction si = m_scene->intersect(eyeRay);
      if (si.hitInformation.hit) {
        if (si.material.Le() != glm::vec3(0.0f)) { // Hit on light source
          L = si.material.Le() / (float)m_numSamples;
        }
        else {
          glm::vec3 tangentSpaceDirection = m_sampler->cosineSampleHemisphere();
          // Construct tangent space
          glm::vec3 notN = glm::normalize(glm::vec3(si.hitInformation.normal.x + 1.0f, si.hitInformation.normal.x + 2.0f, si.hitInformation.normal.x + 3.0f));
          glm::vec3 tangent = glm::normalize(glm::cross(notN, si.hitInformation.normal));
          glm::vec3 bitangent = glm::normalize(glm::cross(si.hitInformation.normal, tangent));

          glm::mat4 tangentToWorld(glm::vec4(tangent, 0.0f), glm::vec4(bitangent, 0.0f), glm::vec4(si.hitInformation.normal, 0.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
          glm::mat4 worldToTangent = glm::inverse(tangentToWorld);
          glm::vec3 worldSpaceDirection = glm::normalize(glm::vec3(tangentToWorld * glm::vec4(tangentSpaceDirection, 0.0f)));

          Ray shadowRay = Ray(si.hitInformation.pos + 1.0e-6f * si.hitInformation.normal, worldSpaceDirection);
          SSurfaceInteraction si2 = m_scene->intersect(shadowRay);

          Ray eyeRayTangent = eyeRay.transform(worldToTangent);

          glm::vec3 f = si.material.f(-eyeRayTangent.m_direction, tangentSpaceDirection);
          glm::vec3 Le = si2.material.Le();
          float cosine = glm::max(glm::dot(si.hitInformation.normal, shadowRay.m_direction), 0.0f);
          float pdf = m_sampler->cosineHemispherePdf(cosine);

          L = f * Le * cosine / ((float)m_numSamples * pdf);
        }
      }
    }
    else if (strategy == IMPORTANCE_SAMPLE_BRDF) {
      Ray eyeRay = m_pixelSampler->samplePixel();

      SSurfaceInteraction si = m_scene->intersect(eyeRay);
      if (si.hitInformation.hit) {
        if (si.material.Le() != glm::vec3(0.0f)) { // Hit on light source
          L = si.material.Le() / (float)m_numSamples;
        }
        else {
          // Construct tangent space
          glm::vec3 notN = glm::normalize(glm::vec3(si.hitInformation.normal.x + 1.0f, si.hitInformation.normal.x + 2.0f, si.hitInformation.normal.x + 3.0f));
          glm::vec3 tangent = glm::normalize(glm::cross(notN, si.hitInformation.normal));
          glm::vec3 bitangent = glm::normalize(glm::cross(si.hitInformation.normal, tangent));

          glm::mat4 tangentToWorld(glm::vec4(tangent, 0.0f), glm::vec4(bitangent, 0.0f), glm::vec4(si.hitInformation.normal, 0.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
          glm::mat4 worldToTangent = glm::inverse(tangentToWorld);


          Ray eyeRayTangent = eyeRay.transform(worldToTangent);
          glm::vec3 wi;
          float pdf;
          //glm::vec3 f = si.material.sampleF(-eyeRayTangent.m_direction, &wi, *m_sampler, &pdf);
          si.material.sampleF(-eyeRayTangent.m_direction, &wi, *m_sampler, &pdf);
          glm::vec3 f = si.material.f(-eyeRayTangent.m_direction, wi);
          
          glm::vec3 worldSpaceDirection = glm::normalize(glm::vec3(tangentToWorld * glm::vec4(wi, 0.0f)));
          Ray shadowRay = Ray(si.hitInformation.pos + 1.0e-6f * si.hitInformation.normal, worldSpaceDirection);
          SSurfaceInteraction si2 = m_scene->intersect(shadowRay);

          glm::vec3 Le = si2.material.Le();
          float cosine = glm::max(glm::dot(si.hitInformation.normal, shadowRay.m_direction), 0.0f);

          L = f * Le * cosine / ((float)m_numSamples * pdf);
        }
      }
    }
    else if (strategy == MULTIPLE_IMPORTANCE_SAMPLE) {
      Ray eyeRay = m_pixelSampler->samplePixel();

      SSurfaceInteraction si = m_scene->intersect(eyeRay);
      if (si.hitInformation.hit) {
        if (si.material.Le() != glm::vec3(0.0f)) { // Hit on light source
          L = si.material.Le() / (float)m_numSamples;
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
          glm::vec3 lightWorldSpaceDirection = glm::normalize(m_scene->sampleLightSources(*m_sampler, &lightsPdf) - si.hitInformation.pos);
          glm::vec3 lightTangentSpaceDirection = glm::normalize(glm::vec3(worldToTangent * glm::vec4(lightWorldSpaceDirection, 0.0f)));

          Ray shadowRayLight = Ray(si.hitInformation.pos + 1.0e-6f * si.hitInformation.normal, lightWorldSpaceDirection);
          SSurfaceInteraction si2 = m_scene->intersect(shadowRayLight);

          Ray eyeRayTangent = eyeRay.transform(worldToTangent);

          float lightSamplingPdf = m_scene->lightSourcePdf(si2, shadowRayLight);
          float lightPdf = lightsPdf * lightSamplingPdf;
          float brdfPdf = si.material.pdf(-eyeRayTangent.m_direction, lightTangentSpaceDirection);
          float mis_weight = balanceHeuristic(1, lightsPdf, 1, brdfPdf);

          glm::vec3 f = si.material.f(-eyeRayTangent.m_direction, lightTangentSpaceDirection);
          glm::vec3 Le = si2.material.Le();
          float distance = glm::length(si.hitInformation.pos - si2.hitInformation.pos);
          float G = glm::max(glm::dot(si.hitInformation.normal, shadowRayLight.m_direction), 0.0f) * glm::max(glm::dot(si2.hitInformation.normal, -shadowRayLight.m_direction), 0.0f) / (distance * distance);

          L += mis_weight * f * Le * G / ((float)m_numSamples * lightPdf);

          // Sample BRDF
          glm::vec3 wi(0.0f);
          f = si.material.sampleF(-eyeRayTangent.m_direction, &wi, *m_sampler, &brdfPdf);
          glm::vec3 brdfWorldSpaceDirection = glm::normalize(glm::vec3(tangentToWorld * glm::vec4(wi, 0.0f)));
          Ray shadowRayBrdf = Ray(si.hitInformation.pos + 1.0e-6f * si.hitInformation.normal, brdfWorldSpaceDirection);
          si2 = m_scene->intersect(shadowRayBrdf);

          Le = si2.material.Le();
          if (Le != glm::vec3(0.0f)) {
            float cosine = glm::max(glm::dot(si.hitInformation.normal, shadowRayBrdf.m_direction), 0.0f);
            lightSamplingPdf = m_scene->lightSourcePdf(si2, shadowRayBrdf);
            lightsPdf = m_scene->lightSourcesPdf(si2);
            lightPdf = lightSamplingPdf * lightsPdf;
            mis_weight = balanceHeuristic(1, brdfPdf, 1, lightPdf);

            L += mis_weight * f * Le * cosine / ((float)m_numSamples * brdfPdf);
          }
          
        }
      }
    }
    return L;
  }
}