#ifndef PATH_INTEGRATOR_IMPL_HPP
#define PATH_INTEGRATOR_IMPL_HPP
#include "path_integrator.hpp"
#include "sampling/mis.hpp"
#include "integrators/objects.hpp"
#include "camera/pixel_sampler.hpp"
#include "scene/interaction.hpp"
#include "scene/device_scene_impl.hpp"
#include "medium/phase_function_impl.hpp"
#include "material/material.hpp"
#include "utility/debugging.hpp"
namespace rt {
  inline CPathIntegrator::CPathIntegrator(CDeviceScene* scene, CPixelSampler* pixelSampler, CSampler* sampler, uint16_t numSamples, bool useBrickGrid, EDebugMode debugMode) :
    m_scene(scene),
    m_pixelSampler(pixelSampler),
    m_sampler(sampler),
    m_numSamples(numSamples),
    m_useBrickGrid(useBrickGrid),
    m_debugMode(debugMode) {

  }

  D_CALLABLE inline glm::vec3 direct(const SInteraction& si, const CMediumInstance* currentMedium, const glm::vec3& wo, const CDeviceScene& scene, CSampler& sampler, bool useBrickGrid, size_t* numLookups) {
    glm::vec3 L(0.f);

    CCoordinateFrame frame = CCoordinateFrame::fromNormal(si.hitInformation.normal);
    glm::vec3 woTangent = frame.worldToTangent() * wo;
    uint3 launchIdx = optixGetLaunchIndex();

    // Sample light source
    {
      glm::vec3 lightWorldSpaceDirection;
      float lightPdf = 0.f;
      glm::vec3 Le = scene.sampleLightSources(sampler, &lightWorldSpaceDirection, &lightPdf);
      if (lightPdf > 0.f) {
        lightWorldSpaceDirection = glm::normalize(lightWorldSpaceDirection);
        glm::vec3 lightTangentSpaceDirection = glm::normalize(frame.worldToTangent() * lightWorldSpaceDirection);

        CRay rayLight = CRay(si.hitInformation.pos, lightWorldSpaceDirection, CRay::DEFAULT_TMAX, currentMedium);
        rayLight.offsetRayOrigin(si.hitInformation.normal);
        if (si.medium) {
          rayLight = si.medium->moveToVoxelBorder(rayLight);
        }
        glm::vec3 trSecondary;
        SInteraction siLight = scene.intersectTr(rayLight, sampler, &trSecondary, useBrickGrid, numLookups);


        glm::vec3 f(0.f);
        float scatteringPdf = 0.f;
        if (si.material) {
          f = si.material->f(si.hitInformation.tc, woTangent, lightTangentSpaceDirection) * glm::abs(glm::dot(si.hitInformation.normal, rayLight.m_direction));
          scatteringPdf = si.material->pdf(woTangent, lightTangentSpaceDirection);
        }
        else {
          scatteringPdf = si.medium->phase().p(wo, lightWorldSpaceDirection, si.hitInformation.sggxS, si.hitInformation.normal, si.hitInformation.ior, sampler);
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
        f = si.material->sampleF(si.hitInformation.tc, woTangent, &wiTangent, sampler, &scatteringPdf);
        wi = glm::normalize(frame.tangentToWorld() * wiTangent);
        f *= glm::abs(glm::dot(si.hitInformation.normal, wi));
      }
      else {
        float p = si.medium->phase().sampleP(wo, &wi, si.hitInformation.sggxS, si.hitInformation.normal, si.hitInformation.ior, sampler);
        f = glm::vec3(p);
        scatteringPdf = p;
      }

      if (scatteringPdf > 0.f) {
        wi = glm::normalize(wi);
        CRay rayBrdf = CRay(si.hitInformation.pos, wi, CRay::DEFAULT_TMAX, currentMedium);
        rayBrdf.offsetRayOrigin(si.hitInformation.normal);
        if (si.medium) {
          rayBrdf = si.medium->moveToVoxelBorder(rayBrdf);
        }
        glm::vec3 trSecondary;
        SInteraction siBrdf = scene.intersectTr(rayBrdf, sampler, &trSecondary, useBrickGrid, numLookups);

        float lightPdf;
        glm::vec3 Le = scene.le(wi, &lightPdf);

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

  inline glm::vec3 CPathIntegrator::Li() const {
    glm::vec3 L(0.0f);
    glm::vec3 throughput(1.f);
    CRay ray = m_pixelSampler->samplePixel();

    bool isEyeRay = true;
    size_t numLookups = 0;
    for (size_t bounces = 0; bounces < 50; ++bounces) {
      SInteraction si;
      //si = m_scene->intersect(ray);
      m_scene->intersect(ray, &si);
      
      SInteraction mi;
      if (si.medium) { //Bounding box hit
        if (ray.m_medium) { // Ray origin inside bb
          if (si.medium && si.medium != ray.m_medium) { // Workaround for infinite loop
            break;
          }
          throughput *= ray.m_medium->sample(ray, *m_sampler, &mi, m_useBrickGrid, &numLookups);
        }
        else { // Ray origin outside bb
          if (isEyeRay && m_debugMode == EDebugMode::VISUALIZE_LODS) {
            L += mapLODs(si.medium->voxelSizeFiltering());
          }
          CRay mediumRay(si.hitInformation.pos, glm::normalize(ray.m_direction), CRay::DEFAULT_TMAX, si.medium);
          mediumRay.offsetRayOrigin(ray.m_direction);
          //SInteraction siMediumEnd = m_scene->intersect(mediumRay);
          SInteraction siMediumEnd;
          m_scene->intersect(mediumRay, &siMediumEnd);
          if (siMediumEnd.hitInformation.hit) {
            if (siMediumEnd.medium && si.medium != siMediumEnd.medium) { // Workaround for infinite loop
              break; 
            }
            throughput *= mediumRay.m_medium->sample(mediumRay, *m_sampler, &mi, m_useBrickGrid, &numLookups);
          }
        }
        if (any(throughput, 0.f)) {
          break;
        }

      }
      isEyeRay = false;
      

      if (mi.medium) {
        L += throughput * direct(mi, mi.medium, -ray.m_direction, *m_scene, *m_sampler, m_useBrickGrid, &numLookups);
        //L = direct(mi, mi.medium, -ray.m_direction, *m_scene, *m_sampler);
        glm::vec3 wo = -ray.m_direction;
        glm::vec3 wi;
        //break;
        mi.medium->phase().sampleP(wo, &wi, mi.hitInformation.sggxS, mi.hitInformation.normal, mi.hitInformation.ior, *m_sampler);
        isEyeRay = false;
        ray = mi.medium->moveToVoxelBorder(CRay(mi.hitInformation.pos, glm::normalize(wi), CRay::DEFAULT_TMAX, mi.medium).offsetRayOrigin(wi));
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
          ray = CRay(si.hitInformation.pos, glm::normalize(ray.m_direction), CRay::DEFAULT_TMAX, !ray.m_medium ? si.medium : nullptr).offsetRayOrigin(ray.m_direction);
          --bounces;
          continue;
        }


        //direct(si, ray.m_medium, -ray.m_direction, *m_scene, *m_sampler);
        L += direct(si, ray.m_medium, -ray.m_direction, *m_scene, *m_sampler, m_useBrickGrid, &numLookups) * throughput;
        //L = (si.hitInformation.normal + 1.f) * 0.5f;
        //break;

        CCoordinateFrame frame = CCoordinateFrame::fromNormal(si.hitInformation.normal);
        CRay rayTangent = ray.transform(frame.worldToTangent());

        // Sample BRDF
        {
          glm::vec3 wi(0.f);
          float brdfPdf = 0.f;
          glm::vec3 f = si.material->sampleF(si.hitInformation.tc, -rayTangent.m_direction, &wi, *m_sampler, &brdfPdf);
          
          isEyeRay = false;
          if (brdfPdf == 0.f) {
            break;
          }

          glm::vec3 brdfWorldSpaceDirection = glm::normalize(frame.tangentToWorld() * wi);
          CRay rayBrdf = CRay(si.hitInformation.pos, brdfWorldSpaceDirection, CRay::DEFAULT_TMAX, ray.m_medium);
          rayBrdf.offsetRayOrigin(si.hitInformation.normal);
          float cosine = glm::abs(glm::dot(si.hitInformation.normal, rayBrdf.m_direction));

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

    if (m_debugMode == EDebugMode::VISUALIZE_VOLUME_LOOKUPS) {
      L += glm::vec3(0.0001f * numLookups, 0.f, 0.f);
    }

    if (glm::any(glm::isnan(L)) || glm::any(glm::isinf(L))) {
      return glm::vec3(0.f, 0.f, 0.f);
    }
    else {
      return L / (float)m_numSamples;
    }

  }

}
#endif // !PATH_INTEGRATOR_IMPL_HPP
