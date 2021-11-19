#include "integrators/path_integrator.hpp"
#include "sampling/sampler.hpp"
#include "scene/scene.hpp"
#include "camera/pixel_sampler.hpp"
#include <cmath>
#include "sampling/mis.hpp"
#include "scene/interaction.hpp"
#include "shapes/circle.hpp"
#include "shapes/sphere.hpp"
#include "integrators/objects.hpp"
#include "utility/functions.hpp"

namespace rt {
  CPathIntegrator::CPathIntegrator(CDeviceScene* scene, CPixelSampler* pixelSampler, CSampler* sampler, uint16_t numSamples):
    m_scene(scene),
    m_pixelSampler(pixelSampler),
    m_sampler(sampler),
    m_numSamples(numSamples) {

  }

  D_CALLABLE glm::vec3 direct(const SInteraction& si, const glm::vec3& wo, const CDeviceScene& scene, CSampler& sampler) {
    glm::vec3 L(0.f);
    
    CCoordinateFrame frame = CCoordinateFrame::fromNormal(si.hitInformation.normal);
    glm::vec3 woTangent = glm::vec3(frame.worldToTangent() * glm::vec4(wo, 0.0f));

    // Sample light source
    {
      glm::vec3 lightWorldSpaceDirection;
      float lightPdf = 0.f;
      glm::vec3 Le = scene.sampleLightSources(sampler, &lightWorldSpaceDirection, &lightPdf);
      if (lightPdf > 0.f) {
        glm::vec3 lightTangentSpaceDirection = glm::normalize(glm::vec3(frame.worldToTangent() * glm::vec4(lightWorldSpaceDirection, 0.0f)));

        CRay rayLight = CRay(si.hitInformation.pos, lightWorldSpaceDirection);
        rayLight.offsetRayOrigin(si.hitInformation.normal);
        glm::vec3 trSecondary;
        SInteraction siLight = scene.intersectTr(rayLight, sampler, &trSecondary); // TODO: Handle case that second hit is on volume



        glm::vec3 f(0.f);
        float scatteringPdf = 0.f;
        if (si.material) {
          f = si.material->f(woTangent, lightTangentSpaceDirection) * glm::max(glm::dot(si.hitInformation.normal, rayLight.m_direction), 0.0f);
          scatteringPdf = si.material->pdf(woTangent, lightTangentSpaceDirection);
        }
        else {
          const CPhaseFunction& phase = si.medium->phase();
          scatteringPdf = si.medium->phase().p(wo, lightWorldSpaceDirection);
          f = glm::vec3(scatteringPdf);
        }

        float mis_weight = balanceHeuristic(1, lightPdf, 1, scatteringPdf);

        glm::vec3 tr;
        if (!siLight.hitInformation.hit) {
          tr = trSecondary; // Light From environment map
        }
        else {
          tr = siLight.material && glm::vec3(0.f) != siLight.material->Le() ? glm::vec3(1.f) : glm::vec3(0.f);
        }

        L += mis_weight * f * Le * tr / (lightPdf);
      }
    }

    // Sample BRDF / Phase function
    {
      float scatteringPdf = 0.f;
      glm::vec3 f(0.f);
      glm::vec3 wi(0.f);
      if (si.material) {
        glm::vec3 wiTangent(0.f);
        f = si.material->sampleF(woTangent, &wiTangent, sampler, &scatteringPdf);
        wi = glm::normalize(glm::vec3(frame.tangentToWorld() * glm::vec4(wiTangent, 0.0f)));
        f *= glm::max(glm::dot(si.hitInformation.normal, wi), 0.f);
      }
      else {
        float p = si.medium->phase().sampleP(wo, &wi, sampler);
        f = glm::vec3(p);
        scatteringPdf = p;
      }

      if (scatteringPdf > 0.f) {
        CRay rayBrdf = CRay(si.hitInformation.pos, wi);
        rayBrdf.offsetRayOrigin(si.hitInformation.normal);
        glm::vec3 trSecondary;
        SInteraction siBrdf = scene.intersectTr(rayBrdf, sampler, &trSecondary);

        float lightPdf;
        glm::vec3 Le = scene.le(wi, &lightPdf);

        float cosine = glm::max(glm::dot(si.hitInformation.normal, rayBrdf.m_direction), 0.0f);
        float mis_weight = balanceHeuristic(1, scatteringPdf, 1, lightPdf);
        glm::vec3 tr;
        if (!siBrdf.hitInformation.hit) {
          tr = trSecondary;
        }
        else {
          tr = siBrdf.material && glm::vec3(0.f) != siBrdf.material->Le() ? glm::vec3(1.f) : glm::vec3(0.f);
        }
        L += mis_weight * f * Le * tr / scatteringPdf;
      }
    }



    return L;
  }

  glm::vec3 CPathIntegrator::Li() const {
    glm::vec3 L(0.0f);
    glm::vec3 throughput(1.f);
    CRay ray = m_pixelSampler->samplePixel();

    bool isEyeRay = true;
    SInteraction si;
    for (size_t bounces = 0; bounces < 10; ++bounces) {
      si = m_scene->intersect(ray);

      SInteraction mi;
      if (si.medium) {
        CRay mediumRay(si.hitInformation.pos, ray.m_direction);
        mediumRay.offsetRayOrigin(ray.m_direction);
        SInteraction siMediumEnd = m_scene->intersect(mediumRay);
        if (siMediumEnd.hitInformation.hit && siMediumEnd.medium) {
          throughput *= si.medium->sample(mediumRay, *m_sampler, &mi);
        }
        if (any(throughput, 0.f)) {
          break;
        }

      }

      if (mi.medium) {
        L += throughput * direct(mi, -ray.m_direction, *m_scene, *m_sampler);
        glm::vec3 wo = -ray.m_direction;
        glm::vec3 wi;
        mi.medium->phase().sampleP(wo, &wi, *m_sampler);
        ray = CRay(mi.hitInformation.pos, wi).offsetRayOrigin(wi);
      }
      else {
        if (bounces == 0) {
          if (!si.hitInformation.hit) {
            float p;
            L += m_scene->le(ray.m_direction, &p) * throughput;
          }
        }

        if (!si.hitInformation.hit) {
          break;
        }

        if (!si.material) {
          ray = CRay(si.hitInformation.pos, ray.m_direction).offsetRayOrigin(ray.m_direction);
          --bounces;
          continue;
        }

        L += direct(si, -ray.m_direction, *m_scene, *m_sampler) * throughput;

        CCoordinateFrame frame = CCoordinateFrame::fromNormal(si.hitInformation.normal);
        CRay rayTangent = ray.transform(frame.worldToTangent());

        // Sample BRDF
        {
          glm::vec3 wi(0.f);
          float brdfPdf = 0.f;
          glm::vec3 f = si.material->sampleF(-rayTangent.m_direction, &wi, *m_sampler, &brdfPdf);
          if (brdfPdf == 0.f) {
            break;
          }

          glm::vec3 brdfWorldSpaceDirection = glm::normalize(glm::vec3(frame.tangentToWorld() * glm::vec4(wi, 0.0f)));
          CRay rayBrdf = CRay(si.hitInformation.pos, brdfWorldSpaceDirection);
          rayBrdf.offsetRayOrigin(si.hitInformation.normal);
          float cosine = glm::max(glm::dot(si.hitInformation.normal, rayBrdf.m_direction), 0.0f);

          throughput *= f * cosine / (brdfPdf);

          ray = rayBrdf;
        }
      }

      if (bounces > 3) {
        float p = 1.f - (throughput.r + throughput.g + throughput.b) / 3.f;
        if (m_sampler->uniformSample01() <= p) {
          break;
        }

        throughput /= (1 - p);
      }

    }

    return L / (float) m_numSamples;
  }
}