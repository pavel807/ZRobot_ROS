/**
 * @file ldlidar_driver.h
 * @author LDRobot (support@ldrobot.com)
 * @brief LD LiDAR Driver wrapper
 * @version 1.0
 * @date 2021-10-28
 */

#ifndef __LDLIDAR_DRIVER_H
#define __LDLIDAR_DRIVER_H

#include "ldlidar_datatype.h"
#include "lipkg.h"
#include "tofbf.h"

#include <string>
#include <memory>

namespace ldlidar {

class LDLidarDriver {
public:
    LDLidarDriver();
    ~LDLidarDriver();

    bool Start(LDType type, const std::string& port, int baudrate, int comm_mode);
    void Stop();
    bool GetLaserScanData(Points2D& out);
    LidarStatus GetLidarStatus();
    void EnableFilter(bool enable);

private:
    LiPkg* pkg_;
    bool is_start_;
};

}

#endif
