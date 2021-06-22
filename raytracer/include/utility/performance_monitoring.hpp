#ifndef PERFORMANCE_MONITORING_HPP
#define PERFORMANCE_MONITORING_HPP

#define TIME_MONITORING

#include <chrono>
#include <unordered_map>
#include <map>
#include <string>
#include <sstream>
#include <cmath>
#include <unordered_set>
#include <algorithm>

namespace rt {
  class CPerformanceMonitoring {

    CPerformanceMonitoring() = delete;

  public:
    static void startMeasurement(const std::string&& name) {
#ifdef TIME_MONITORING
      auto time = s_clock.now();
      if (s_currentMeasurements.find(name) == s_currentMeasurements.end()) {
        s_currentMeasurements[name] = time;
      }
#endif // TIME_MONITORING
    }

    static void endMeasurement(const std::string&& name) {
#ifdef TIME_MONITORING
      auto time = s_clock.now();
      if (s_currentMeasurements.find(name) != s_currentMeasurements.end()) {
        std::chrono::duration<double> diff = time - s_currentMeasurements[name];
        s_durations.insert({ name, diff.count() });
        s_currentMeasurements.erase(name);
      }
#endif // TIME_MONITORING
    }

    static std::string toString() {
      std::string output = "";
#ifdef TIME_MONITORING
      std::unordered_set<std::string> keys;
      for (auto it = s_durations.cbegin(); it != s_durations.cend(); ++it) {
        keys.insert(it->first);
      }

      std::ostringstream out;
      out.precision(10);
      for (auto& key : keys) {
        auto range = s_durations.equal_range(key);
        std::vector<double> values = sort(range.first, range.second);
        double a = avg(range.first, range.second);
        double s = stdDev(range.first, range.second, a);
        double minimum = min(values);
        double first_quartile = firstQuartile(values);
        double median_ = median(values);
        double third_quartile = thirdQuartile(values);
        double maximum = max(values);
        out << std::fixed << a << "," << s << "," << minimum << "," << first_quartile << "," << median_ << "," << third_quartile << "," << maximum;
        output += key + "," + out.str() + "\n";
        out.str("");
        out.clear();
      }
#endif // TIME_MONITORING
      return output;
    }


  private:
    static std::chrono::steady_clock s_clock;
    static std::unordered_map<std::string, std::chrono::time_point<std::chrono::steady_clock>> s_currentMeasurements;
    static std::multimap<std::string, double> s_durations;

    static double min(std::vector<double>& sortedValues) {
      return sortedValues[0];
    }

    static double max(std::vector<double>& sortedValues) {
      return sortedValues[sortedValues.size() - 1];
    }

    static double median(std::vector<double>& sortedValues) {
      if (sortedValues.size() % 2 == 1) {
        return sortedValues[sortedValues.size() / 2];
      }
      else {
        return 0.5 * (sortedValues[sortedValues.size() / 2 - 1] + sortedValues[sortedValues.size() / 2]);
      }
      return sortedValues[sortedValues.size() - 1];
    }

    static double firstQuartile(std::vector<double>& sortedValues) {
      if (sortedValues.size() % 4 == 1) {
        return sortedValues[sortedValues.size() / 4];
      }
      else {
        double weight = sortedValues.size() / 4.0;
        size_t lowerIndex = weight;
        size_t upperIndex = lowerIndex + 1;
        return (upperIndex - weight) * sortedValues[lowerIndex] + (weight - lowerIndex) * sortedValues[upperIndex];
      }
    }

    static double thirdQuartile(std::vector<double>& sortedValues) {
      if (sortedValues.size() * 3 % 4 == 1) {
        return sortedValues[sortedValues.size() * 3 / 4];
      }
      else {
        double weight = sortedValues.size() * 3 / 4.0;
        size_t lowerIndex = weight;
        size_t upperIndex = lowerIndex + 1;
        return (upperIndex - weight) * sortedValues[lowerIndex] + (weight - lowerIndex) * sortedValues[upperIndex];
      }
    }

    static std::vector<double> sort(std::multimap<std::string, double>::iterator begin, std::multimap<std::string, double>::iterator end) {
      size_t count = std::distance(begin, end);
      std::vector<double> elements;
      elements.reserve(count);
      for (auto& it = begin; it != end; ++it) {
        elements.push_back(it->second);
      }
      std::sort(elements.begin(), elements.end());
      return elements;
    }

    static double avg(std::multimap<std::string, double>::iterator begin, std::multimap<std::string, double>::iterator end) {
      double average = 0.0f;
      size_t count = 0;
      for (auto& it = begin; it != end; ++it) {
        average += it->second;
        ++count;
      }
      return average /= count;
    }

    static double stdDev(std::multimap<std::string, double>::iterator begin, std::multimap<std::string, double>::iterator end, double avg) {
      double variance = 0.0f;
      size_t count = 0;
      for (auto& it = begin; it != end; ++it) {
        double diff = it->second - avg;
        variance += diff * diff;
        ++count;
      }
      return std::sqrt(variance / (count - 1));
    }
  };

}
#endif // !PERFORMANCE_MONITORING_HPP
