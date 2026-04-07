/**
 * @file ldlidar_driver.cpp
 * @author LDRobot (support@ldrobot.com)
 * @brief LD LiDAR Driver implementation
 * @version 1.0
 * @date 2021-10-28
 */

#include "ldlidar_driver.h"
#include <chrono>
#include <thread>

namespace ldlidar {

LDLidarDriver::LDLidarDriver() 
    : pkg_(nullptr), tofbf_(nullptr), is_start_(false), filter_enabled_(false), lidar_freq_(10.0) {
    pkg_ = new LiPkg();
}

LDLidarDriver::~LDLidarDriver() {
    Stop();
    if (pkg_) {
        delete pkg_;
        pkg_ = nullptr;
    }
    if (tofbf_) {
        delete tofbf_;
        tofbf_ = nullptr;
    }
}

std::string LDLidarDriver::GetLidarSdkVersionNumber() {
    return "V1.0.0";
}

void LDLidarDriver::RegisterGetTimestampFunctional(std::function<uint64_t(void)> timestamp_handle) {
    if (pkg_) {
        pkg_->RegisterTimestampGetFunctional(timestamp_handle);
    }
}

void LDLidarDriver::EnableFilterAlgorithnmProcess(bool enable) {
    filter_enabled_ = enable;
    if (pkg_) {
        pkg_->EnableFilter(enable);
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

bool LDLidarDriver::WaitLidarCommConnect(int timeout_ms) {
    if (!pkg_ || !is_start_) return false;
    
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count() < timeout_ms) {
        
        if (pkg_->GetLidarPowerOnCommStatus()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
}

void LDLidarDriver::Stop() {
    is_start_ = false;
    if (pkg_) {
        pkg_->ClearDataProcessStatus();
    }
}

LidarStatus LDLidarDriver::GetLaserScanData(Points2D& out, int timeout_ms) {
    (void)timeout_ms;
    
    if (pkg_ && is_start_) {
        if (pkg_->GetLaserScanData(out)) {
            if (filter_enabled_ && tofbf_) {
                out = tofbf_->Filter(out);
            }
            return LidarStatus::NORMAL;
        }
        
        auto status = pkg_->GetLidarStatus();
        if (status == LidarStatus::NORMAL) {
            return LidarStatus::DATA_WAIT;
        }
        return status;
    }
    return LidarStatus::DATA_TIME_OUT;
}

void LDLidarDriver::GetLidarScanFreq(double& freq) {
    if (pkg_) {
        lidar_freq_ = pkg_->GetSpeed();
    }
    freq = lidar_freq_;
}

LidarStatus LDLidarDriver::GetLidarStatus() {
    if (pkg_) {
        return pkg_->GetLidarStatus();
    }
    return LidarStatus::HARD_ERROR;
}

void LDLidarDriver::EnableFilter(bool enable) {
    EnableFilterAlgorithnmProcess(enable);
}

}
