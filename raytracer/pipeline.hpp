#ifndef PIPELINE_HPP
#define PIPELINE_HPP
#include "backend/config_loader.hpp"

class CPipeline {
public:
  CPipeline(const char* configPath);
  void run();


private:
  SConfig m_config;
  
  void allocateDeviceMemory();
  void copyToDevice();
  void freeDeviceMemory();
};
#endif