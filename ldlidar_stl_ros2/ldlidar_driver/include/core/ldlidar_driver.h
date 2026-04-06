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

class LiDARDriver {
public:
    LiDARDriver();
    ~LiDARDriver();

    bool init(LDType type, const std::string& port, int baudrate);
    bool start();
    void stop();
    bool getScanData(Points2D& points);
    LidarStatus getStatus() const;
    void enableFilter(bool enable);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}

#endif
