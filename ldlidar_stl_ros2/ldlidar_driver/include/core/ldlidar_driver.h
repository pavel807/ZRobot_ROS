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
#include <functional>

namespace ldlidar {

class LDLidarDriver {
public:
    LDLidarDriver();
    ~LDLidarDriver();

    std::string GetLidarSdkVersionNumber();
    
    void RegisterGetTimestampFunctional(std::function<uint64_t(void)> timestamp_handle);
    
    void EnableFilterAlgorithnmProcess(bool enable);
    
    bool Start(LDType type, const std::string& port, int baudrate, int comm_mode);
    
    bool WaitLidarCommConnect(int timeout_ms);
    
    void Stop();
    
    LidarStatus GetLaserScanData(Points2D& out, int timeout_ms);
    
    void GetLidarScanFreq(double& freq);
    
    LidarStatus GetLidarStatus();
    
    void EnableFilter(bool enable);

private:
    LiPkg* pkg_;
    Tofbf* tofbf_;
    bool is_start_;
    bool filter_enabled_;
    double lidar_freq_;
};

}

#endif
