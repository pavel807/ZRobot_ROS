/**
 * @file ldlidar_datatype.h
 * @author LDRobot (support@ldrobot.com)
 * @brief LD LiDAR data type definitions
 * @version 1.0
 * @date 2021-10-28
 */

#ifndef __LDLIDAR_DATATYPE_H
#define __LDLIDAR_DATATYPE_H

#include <vector>
#include <string>

namespace ldlidar {

enum class LDType {
  NO_VERSION = 0,
  LD_06 = 1,
  LD_19 = 2,
  STL_06P = 3,
  STL_26 = 4,
  STL_27L = 5,
};

enum class LidarStatus {
  NORMAL = 0,
  DATA_TIME_OUT = 1,
  DATA_WAIT = 2,
  HARD_ERROR = 3,
};

struct PointData {
  float distance;
  float angle;
  uint8_t intensity;
  
  PointData() : distance(0.0f), angle(0.0f), intensity(0) {}
  PointData(float d, float a, uint8_t i) : distance(d), angle(a), intensity(i) {}
};

typedef std::vector<PointData> Points2D;

struct LidarPoint {
  float x;
  float y;
  float z;
  
  LidarPoint() : x(0), y(0), z(0) {}
  LidarPoint(float px, float py, float pz) : x(px), y(py), z(pz) {}
};

typedef std::vector<LidarPoint> Points3D;

}

#endif
