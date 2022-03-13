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

  H_CALLABLE inline bool getMask(const nlohmann::json& jArray, rt::ESceneobjectMask* mask) {
    bool valid = !jArray.empty();
    uint8_t m = rt::ESceneobjectMask::NONE;
    for (auto& item : jArray) {
      m |= rt::getMask(item.get<std::string>());
    }
    *mask = static_cast<rt::ESceneobjectMask>(m);
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
    auto samplesPerVoxel = filtering["SamplesPerVoxel"];
    glm::vec3 voxelSize;
    auto debug = filtering["Debug"];
    auto debugSamples = filtering["DebugSamples"];
    auto sigmaT = filtering["SigmaT"];
    auto estimationIterations = filtering["EstimationIterations"];
    auto alpha = filtering["Alpha"];
    auto clipRays = filtering["ClipRays"];
    bool valid = !filtering.empty() && parseVec3(filtering["VoxelSize"], &voxelSize) && !filter.empty() && !samplesPerVoxel.empty() && !debug.empty() && !debugSamples.empty() && !sigmaT.empty() && !estimationIterations.empty() && !alpha.empty() && !clipRays.empty();
    if (valid) {
      config->filteringConfig.filter = filter.get<bool>();
      config->filteringConfig.voxelSize = voxelSize;
      config->filteringConfig.samplesPerVoxel = samplesPerVoxel.get<uint32_t>();
      config->filteringConfig.debug = debug.get<bool>();
      config->filteringConfig.debugSamples = debugSamples.get<uint32_t>();
      config->filteringConfig.sigmaT = sigmaT.get<float>();
      config->filteringConfig.estimationIterations = estimationIterations.get<uint32_t>();
      config->filteringConfig.alpha = alpha.get<float>();
      config->filteringConfig.clipRays = clipRays.get<bool>();
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
      config->camera = std::make_shared<rt::CCamera>(config->frameWidth,
        config->frameHeight,
        fov.get<float>(),
        pos,
        lookAt,
        up);
    }

    return valid;
  }

  H_CALLABLE bool addCircle(const nlohmann::json& sceneobject, rt::CHostScene* scene) {
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
      auto maskArray = sceneobject.find("Mask");
      rt::ESceneobjectMask mask = rt::ESceneobjectMask::RENDER;
      if (maskArray != sceneobject.end()) {
        getMask(sceneobject["Mask"], &mask);
      }

      scene->addSceneobject(rt::CHostSceneobject(
        new rt::CCircle(pos, radius.get<float>(), normal),
        diffuseReflection,
        diffuseRoughness.get<float>(),
        specularReflection,
        alphaX.get<float>(),
        alphaY.get<float>(),
        etaI.get<float>(),
        etaT.get<float>(),
        mask));
    }
    return valid;
  }

  H_CALLABLE bool addSphere(const nlohmann::json& sceneobject, rt::CHostScene* scene) {
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
      auto maskArray = sceneobject.find("Mask");
      rt::ESceneobjectMask mask = rt::ESceneobjectMask::RENDER;
      if (maskArray != sceneobject.end()) {
        getMask(sceneobject["Mask"], &mask);
      }

      scene->addSceneobject(rt::CHostSceneobject(
        new rt::Sphere(pos, radius.get<float>(), normal),
        diffuseReflection,
        diffuseRoughness.get<float>(),
        specularReflection,
        alphaX.get<float>(),
        alphaY.get<float>(),
        etaI.get<float>(),
        etaT.get<float>(),
        mask));
    }
    return valid;
  }

  H_CALLABLE bool addMedium(const nlohmann::json& sceneobject, rt::CHostScene* scene) {
    bool valid = true;

    auto path = sceneobject["Path"];
    valid = valid && !path.empty();

    glm::vec3 sigmaA;
    valid = valid && parseVec3(sceneobject["SigmaA"], &sigmaA);

    glm::vec3 sigmaS;
    valid = valid && parseVec3(sceneobject["SigmaS"], &sigmaS);

    auto g = sceneobject.find("G");

    glm::vec3 pos;
    valid = valid && parseVec3(sceneobject["Pos"], &pos);

    glm::vec3 orientation;
    valid = valid && parseVec3(sceneobject["Orientation"], &orientation);

    glm::vec3 scaling;
    valid = valid && parseVec3(sceneobject["Scaling"], &scaling);

    if (valid) {
      auto maskArray = sceneobject.find("Mask");
      rt::ESceneobjectMask mask = rt::ESceneobjectMask::RENDER;
      if (maskArray != sceneobject.end()) {
        getMask(sceneobject["Mask"], &mask);
      }

      rt::CNVDBMedium* medium = nullptr;
      if (g == sceneobject.end()) {
        medium = rt::CAssetManager::loadMedium(
          path.get<std::string>(),
          sigmaA,
          sigmaS);
      }
      else {
        medium = rt::CAssetManager::loadMedium(
          path.get<std::string>(),
          sigmaA,
          sigmaS,
          g.value().get<float>());
      }
      scene->addSceneobject(rt::CHostSceneobject(
        medium,
        pos,
        orientation,
        scaling,
        mask));
    }
    return valid;
  }

  H_CALLABLE bool addMesh(const nlohmann::json& sceneobject, rt::CHostScene* scene) {
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
      auto maskArray = sceneobject.find("Mask");
      rt::ESceneobjectMask mask = rt::ESceneobjectMask::RENDER;
      if (maskArray != sceneobject.end()) {
        getMask(sceneobject["Mask"], &mask);
      }

      scene->addSceneobjectsFromAssimp(
        directory.get<std::string>(),
        filename.get<std::string>(),
        pos,
        orientation,
        scaling,
        mask);
    }
    return valid;
  }

  H_CALLABLE bool fillSceneInfo(const nlohmann::json& scene, SConfig* config) {
    config->scene = std::make_shared<rt::CHostScene>();

    bool valid = true;
    auto envmap = scene["Envmap"];

    valid = valid && !scene.empty() && !envmap.empty();
    if (valid) {
      config->scene->setEnvironmentMap(rt::CEnvironmentMap(envmap.get<std::string>()));
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
