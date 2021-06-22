#include "utility/performance_monitoring.hpp"

namespace rt {
  std::chrono::steady_clock CPerformanceMonitoring::s_clock;
  std::unordered_map<std::string, std::chrono::time_point<std::chrono::steady_clock>> CPerformanceMonitoring::s_currentMeasurements;
  std::multimap<std::string, double> CPerformanceMonitoring::s_durations;
}