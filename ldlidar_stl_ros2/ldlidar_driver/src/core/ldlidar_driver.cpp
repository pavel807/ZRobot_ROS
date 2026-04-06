/**
 * @file ldlidar_driver.cpp
 * @author LDRobot (support@ldrobot.com)
 * @brief LD LiDAR Driver implementation
 * @version 1.0
 * @date 2021-10-28
 */

#include "ldlidar_driver.h"

namespace ldlidar {

LDLidarDriver::LDLidarDriver() : pkg_(nullptr), is_start_(false) {
    pkg_ = new LiPkg();
}

LDLidarDriver::~LDLidarDriver() {
    Stop();
    if (pkg_) {
        delete pkg_;
        pkg_ = nullptr;
    }
}

bool LDLidarDriver::Start(LDType type, const std::string& port, int baudrate, int comm_mode) {
    (void)port;
    (void)baudrate;
    (void)comm_mode;
    
    if (pkg_) {
        pkg_->SetProductType(type);
        is_start_ = true;
        return true;
    }
    return false;
}

void LDLidarDriver::Stop() {
    is_start_ = false;
    if (pkg_) {
        pkg_->ClearDataProcessStatus();
    }
}

bool LDLidarDriver::GetLaserScanData(Points2D& out) {
    if (pkg_ && is_start_) {
        return pkg_->GetLaserScanData(out);
    }
    return false;
}

LidarStatus LDLidarDriver::GetLidarStatus() {
    if (pkg_) {
        return pkg_->GetLidarStatus();
    }
    return LidarStatus::HARD_ERROR;
}

void LDLidarDriver::EnableFilter(bool enable) {
    if (pkg_) {
        pkg_->EnableFilter(enable);
    }
}

}
