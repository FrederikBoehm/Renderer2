#include "integrators/path_integrator.hpp"
#include "sampling/sampler.hpp"
#include "scene/scene.hpp"
#include "camera/pixel_sampler.hpp"
#include <cmath>
#include "sampling/mis.hpp"
#include "scene/interaction.hpp"
#include "shapes/plane.hpp"
#include "shapes/sphere.hpp"
#include "integrators/objects.hpp"

namespace rt {
  CPathIntegrator::CPathIntegrator(CDeviceScene* scene, CPixelSampler* pixelSampler, CSampler* sampler, uint16_t numSamples):
    m_scene(scene),
    m_pixelSampler(pixelSampler),
    m_sampler(sampler),
    m_numSamples(numSamples) {

  }

  D_CALLABLE glm::vec3 direct(const SInteraction& si, const glm::vec3& wo, const CCoordinateFrame& frame, const CDeviceScene& scene, CSampler& sampler) {
    

    glm::vec3 L(0.f);
    // Sample light source
    {
      float lightsPdf = 0.0f;
      glm::vec3 lightWorldSpaceDirection;
      float lightPdf;
      glm::vec3 Le = scene.sampleLightSources(sampler, &lightWorldSpaceDirection, &lightPdf);
      glm::vec3 lightTangentSpaceDirection = glm::normalize(glm::vec3(frame.worldToTangent() * glm::vec4(lightWorldSpaceDirection, 0.0f)));

      CRay rayLight = CRay(si.hitInformation.pos + 1.0e-6f * si.hitInformation.normal, lightWorldSpaceDirection);
      glm::vec3 trSecondary;
      SInteraction siLight = scene.intersectTr(rayLight, sampler, &trSecondary); // TODO: Handle case that second hit is on volume


      float brdfPdf = si.material->pdf(wo, lightTangentSpaceDirection);
      float mis_weight = balanceHeuristic(1, lightPdf, 1, brdfPdf);

      glm::vec3 f = si.material->f(wo, lightTangentSpaceDirection);
      glm::vec3 tr;
      if (!siLight.hitInformation.hit) {
        tr = trSecondary; // Light From environment map
      }
      else {
        tr = siLight.material && glm::vec3(0.f) != siLight.material->Le() ? glm::vec3(1.f) : glm::vec3(0.f);
      }

      if (lightPdf > 0.f) {
        L += mis_weight * f * Le * glm::max(glm::dot(si.hitInformation.normal, rayLight.m_direction), 0.0f) * tr / (lightPdf);
        //L += tr;
      }
    }

    // Sample BRDF
    {
      glm::vec3 wi(0.f);
      float brdfPdf = 0.f;
      glm::vec3 f = si.material->sampleF(wo, &wi, sampler, &brdfPdf);

      glm::vec3 brdfWorldSpaceDirection = glm::normalize(glm::vec3(frame.tangentToWorld() * glm::vec4(wi, 0.0f)));
      CRay rayBrdf = CRay(si.hitInformation.pos + 1.0e-6f * si.hitInformation.normal, brdfWorldSpaceDirection);
      glm::vec3 trSecondary;
      SInteraction siBrdf = scene.intersectTr(rayBrdf, sampler, &trSecondary);

      float lightPdf;
      glm::vec3 Le = scene.le(brdfWorldSpaceDirection, &lightPdf);

      float cosine = glm::max(glm::dot(si.hitInformation.normal, rayBrdf.m_direction), 0.0f);
      float mis_weight = balanceHeuristic(1, brdfPdf, 1, lightPdf);
      glm::vec3 tr;
      if (!siBrdf.hitInformation.hit) {
        tr = trSecondary;
      }
      else {
        tr = siBrdf.material && glm::vec3(0.f) != siBrdf.material->Le() ? glm::vec3(1.f) : glm::vec3(0.f);
      }
      if (brdfPdf > 0.f) {
        L += mis_weight * f * Le * glm::max(glm::dot(si.hitInformation.normal, rayBrdf.m_direction), 0.f) * tr / brdfPdf;
      }
    }

    return L;
  }

  glm::vec3 CPathIntegrator::Li() const {
    glm::vec3 L(0.0f);
    glm::vec3 throughput(1.f);
    CRay ray = m_pixelSampler->samplePixel();

    bool isEyeRay = true;
    SInteraction si = m_scene->intersect(ray);
    size_t numBounces = 0;
    for (size_t bounces = 0; bounces < 100; ++bounces) {
      numBounces = bounces;
      if (isEyeRay) {
        if (si.hitInformation.hit && si.medium) {
          glm::vec3 tr(1.f);
          si = m_scene->intersectTr(CRay(si.hitInformation.pos + 1e-6f * ray.m_direction, ray.m_direction), *m_sampler, &tr);
          throughput *= tr;
        }
        if (!si.hitInformation.hit) {
          float p;
          L += m_scene->le(ray.m_direction, &p) * throughput;
          break;
        }
        isEyeRay = false;
      }
      if (!si.hitInformation.hit) {
        break;
      }


      CCoordinateFrame frame = CCoordinateFrame::fromNormal(si.hitInformation.normal);
      CRay rayTangent = ray.transform(frame.worldToTangent());

      L += direct(si, -rayTangent.m_direction, frame, *m_scene, *m_sampler) * throughput;

      float p = 1.f - (throughput.r + throughput.g + throughput.b) / 3.f;
      if (m_sampler->uniformSample01() <= p) {
        break;
      }

      // Sample BRDF
      {
        glm::vec3 wi(0.f);
        float brdfPdf = 0.f;
        glm::vec3 f = si.material->sampleF(-rayTangent.m_direction, &wi, *m_sampler, &brdfPdf);
        if (brdfPdf == 0.f) {
          break;
        }
        
        glm::vec3 brdfWorldSpaceDirection = glm::normalize(glm::vec3(frame.tangentToWorld() * glm::vec4(wi, 0.0f)));
        CRay rayBrdf = CRay(si.hitInformation.pos + 1.0e-6f * si.hitInformation.normal, brdfWorldSpaceDirection);
        glm::vec3 trSecondary;
        SInteraction siBrdf = m_scene->intersectTr(rayBrdf, *m_sampler, &trSecondary);


        float cosine = glm::max(glm::dot(si.hitInformation.normal, rayBrdf.m_direction), 0.0f);

        throughput *= f * cosine * trSecondary / (brdfPdf * (1.f - p));

        si = siBrdf;
        ray = rayBrdf;
      }
    }
    return L / (float) m_numSamples;
  }
}