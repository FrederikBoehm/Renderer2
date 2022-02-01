#include "backend/config_loader.hpp"
#include <json/json.hpp>
#include <fstream>
#include <sstream>
#include <string>
#include "scene/environmentmap.hpp"
#include "medium/nvdb_medium.hpp"
#include <exception>
#include "backend/asset_manager.hpp"
#include "scene/sceneobject_mask.hpp"

namespace rt {
  H_CALLABLE inline bool parseVec3(const nlohmann::json& jArray, glm::vec3* outVec) {
    bool valid = true;
    valid = valid && !jArray.empty() && jArray.size() == 3 && outVec;
    if (valid) {
      outVec->x = jArray[0].get<float>();
      outVec->y = jArray[1].get<float>();
      outVec->z = jArray[2].get<float>();
    }
    return valid;
  }

  H_CALLABLE inline bool parseVec3(const nlohmann::json& jArray, glm::ivec3* outVec) {
    bool valid = true;
    valid = valid && !jArray.empty() && jArray.size() == 3 && outVec;
    if (valid) {
      outVec->x = jArray[0].get<int>();
      outVec->y = jArray[1].get<int>();
      outVec->z = jArray[2].get<int>();
    }
    return valid;
  }

  H_CALLABLE bool fillGeneralInfo(const nlohmann::json& general, SConfig* config) {
    auto frameWidth = general["FrameWidth"];
    auto frameHeight = general["FrameHeight"];
    auto channelsPerPixel = general["ChannelsPerPixel"];
    auto samples = general["Samples"];
    auto gamma = general["Gamma"];

    bool valid = !general.empty() && !frameWidth.empty() && !frameHeight.empty() && !channelsPerPixel.empty() && !samples.empty() && !gamma.empty();
    if (valid) {
      config->frameWidth = frameWidth.get<uint16_t>();
      config->frameHeight = frameHeight.get<uint16_t>();
      config->channelsPerPixel = channelsPerPixel.get<uint8_t>();
      config->samples = samples.get<uint16_t>();
      config->gamma = gamma.get<float>();
    }
    return valid;
  }

  H_CALLABLE bool fillFilteringInfo(const nlohmann::json& filtering, SConfig* config) {
    auto filter = filtering["Active"];
    glm::ivec3 numVoxels;
    bool valid = !filtering.empty() && parseVec3(filtering["NumVoxels"], &numVoxels) && !filter.empty();
    if (valid) {
      config->filteringConfig.filter = filter.get<bool>();
      config->filteringConfig.numVoxels = numVoxels;
    }
    return valid;
  }

  H_CALLABLE bool fillCameraInfo(const nlohmann::json& camera, SConfig* config) {
    auto fov = camera["Fov"];

    glm::vec3 pos;
    glm::vec3 lookAt;
    glm::vec3 up;

    bool valid = !camera.empty() && !fov.empty() && parseVec3(camera["Pos"], &pos) && parseVec3(camera["LookAt"], &lookAt) && parseVec3(camera["Up"], &up);

    if (valid) {
      config->camera = std::make_shared<CCamera>(config->frameWidth,
        config->frameHeight,
        fov.get<float>(),
        pos,
        lookAt,
        up);
    }

    return valid;
  }

  H_CALLABLE bool addCircle(const nlohmann::json& sceneobject, CHostScene* scene) {
    bool valid = true;

    glm::vec3 pos;
    valid = valid && parseVec3(sceneobject["Pos"], &pos);

    auto radius = sceneobject["Radius"];
    valid = valid && !radius.empty();

    glm::vec3 normal;
    valid = valid && parseVec3(sceneobject["Normal"], &normal);

    glm::vec3 diffuseReflection;
    valid = valid && parseVec3(sceneobject["DiffuseReflection"], &diffuseReflection);

    auto diffuseRoughness = sceneobject["DiffuseRoughness"];
    valid = valid && !diffuseRoughness.empty();

    glm::vec3 specularReflection;
    valid = valid && parseVec3(sceneobject["SpecularReflection"], &specularReflection);

    auto alphaX = sceneobject["AlphaX"];
    auto alphaY = sceneobject["AlphaY"];
    auto etaI = sceneobject["EtaI"];
    auto etaT = sceneobject["EtaT"];
    valid = valid && !alphaX.empty() && !alphaY.empty() && !etaI.empty() && !etaT.empty();

    if (valid) {
      auto mask = sceneobject.find("Mask");

      scene->addSceneobject(CHostSceneobject(
        new CCircle(pos, radius.get<float>(), normal),
        diffuseReflection,
        diffuseRoughness.get<float>(),
        specularReflection,
        alphaX.get<float>(),
        alphaY.get<float>(),
        etaI.get<float>(),
        etaT.get<float>(),
        mask == sceneobject.end() ? ESceneobjectMask::NONE : getMask(mask->get<std::string>())));
    }
    return valid;
  }

  H_CALLABLE bool addSphere(const nlohmann::json& sceneobject, CHostScene* scene) {
    bool valid = true;

    glm::vec3 pos;
    valid = valid && parseVec3(sceneobject["Pos"], &pos);

    auto radius = sceneobject["Radius"];
    valid = valid && !radius.empty();

    glm::vec3 normal;
    valid = valid && parseVec3(sceneobject["Normal"], &normal);

    glm::vec3 diffuseReflection;
    valid = valid && parseVec3(sceneobject["DiffuseReflection"], &diffuseReflection);

    auto diffuseRoughness = sceneobject["DiffuseRoughness"];
    valid = valid && !diffuseRoughness.empty();

    glm::vec3 specularReflection;
    valid = valid && parseVec3(sceneobject["SpecularReflection"], &specularReflection);

    auto alphaX = sceneobject["AlphaX"];
    auto alphaY = sceneobject["AlphaY"];
    auto etaI = sceneobject["EtaI"];
    auto etaT = sceneobject["EtaT"];
    valid = valid && !alphaX.empty() && !alphaY.empty() && !etaI.empty() && !etaT.empty();

    if (valid) {
      auto mask = sceneobject.find("Mask");

      scene->addSceneobject(CHostSceneobject(
        new Sphere(pos, radius.get<float>(), normal),
        diffuseReflection,
        diffuseRoughness.get<float>(),
        specularReflection,
        alphaX.get<float>(),
        alphaY.get<float>(),
        etaI.get<float>(),
        etaT.get<float>(),
        mask == sceneobject.end() ? ESceneobjectMask::NONE : getMask(mask->get<std::string>())));
    }
    return valid;
  }

  H_CALLABLE bool addMedium(const nlohmann::json& sceneobject, CHostScene* scene) {
    bool valid = true;

    auto path = sceneobject["Path"];
    valid = valid && !path.empty();

    glm::vec3 sigmaA;
    valid = valid && parseVec3(sceneobject["SigmaA"], &sigmaA);

    glm::vec3 sigmaS;
    valid = valid && parseVec3(sceneobject["SigmaS"], &sigmaS);

    auto diffuseRoughness = sceneobject["DiffuseRoughness"];
    auto specularRoughness = sceneobject["SpecularRoughness"];
    valid = valid && !diffuseRoughness.empty() && !specularRoughness.empty();

    glm::vec3 pos;
    valid = valid && parseVec3(sceneobject["Pos"], &pos);

    glm::vec3 orientation;
    valid = valid && parseVec3(sceneobject["Orientation"], &orientation);

    glm::vec3 scaling;
    valid = valid && parseVec3(sceneobject["Scaling"], &scaling);

    if (valid) {
      auto mask = sceneobject.find("Mask");

      CNVDBMedium* medium = CAssetManager::loadMedium(
        path.get<std::string>(),
        sigmaA,
        sigmaS,
        diffuseRoughness.get<float>(),
        specularRoughness.get<float>());
      scene->addSceneobject(CHostSceneobject(
        medium,
        pos,
        orientation,
        scaling,
        mask == sceneobject.end() ? ESceneobjectMask::NONE : getMask(mask->get<std::string>())));
    }
    return valid;
  }

  H_CALLABLE bool addMesh(const nlohmann::json& sceneobject, CHostScene* scene) {
    bool valid = true;

    auto directory = sceneobject["Directory"];
    auto filename = sceneobject["Filename"];
    valid = valid && !directory.empty() && !directory.empty();
    
    glm::vec3 pos;
    valid = valid && parseVec3(sceneobject["Pos"], &pos);

    glm::vec3 orientation;
    valid = valid && parseVec3(sceneobject["Orientation"], &orientation);

    glm::vec3 scaling;
    valid = valid && parseVec3(sceneobject["Scaling"], &scaling);

    if (valid) {
      auto mask = sceneobject.find("Mask");

      scene->addSceneobjectsFromAssimp(
        directory.get<std::string>(),
        filename.get<std::string>(),
        pos,
        orientation,
        scaling,
        mask == sceneobject.end() ? ESceneobjectMask::NONE : getMask(mask->get<std::string>()));
    }
    return valid;
  }

  H_CALLABLE bool fillSceneInfo(const nlohmann::json& scene, SConfig* config) {
    config->scene = std::make_shared<CHostScene>();

    bool valid = true;
    auto envmap = scene["Envmap"];

    valid = valid && !scene.empty() && !envmap.empty();
    if (valid) {
      config->scene->setEnvironmentMap(CEnvironmentMap(envmap.get<std::string>()));
    }

    auto sceneobjects = scene["Sceneobjects"];
    for (auto sceneobject : sceneobjects) {
      auto type = sceneobject["Type"];
      valid = valid && !type.empty();
      if (valid) {
        std::string typeStr = type.get<std::string>();
        if (typeStr == "Circle") {
          valid = valid && addCircle(sceneobject, config->scene.get());
        }
        else if (typeStr == "Sphere") {
          valid = valid && addSphere(sceneobject, config->scene.get());
        }
        else if (typeStr == "Medium") {
          valid = valid && addMedium(sceneobject, config->scene.get());
        }
        else if (typeStr == "Mesh") {
          valid = valid && addMesh(sceneobject, config->scene.get());
        }
      }
    }

    return valid;
  }

  SConfig CConfigLoader::loadConfig(const char* configPath) {
    std::ifstream inputStream(configPath);
    std::stringstream buffer;
    buffer << inputStream.rdbuf();
    std::string jsonString = buffer.str();
    nlohmann::json jConfig = nlohmann::json::parse(jsonString, nullptr, true, true);

    bool valid = true;
    SConfig config;

    valid = valid && fillGeneralInfo(jConfig["General"], &config);
    valid = valid && fillFilteringInfo(jConfig["Filtering"], &config);
    valid = valid && fillCameraInfo(jConfig["Camera"], &config);
    valid = valid && fillSceneInfo(jConfig["Scene"], &config);
    if (!valid) {
      throw new std::exception("Could not parse config\n");
    }

    return config;
  }
}