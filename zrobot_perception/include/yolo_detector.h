#ifndef YOLO_DETECTOR_H
#define YOLO_DETECTOR_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <rknn_api.h>
#include "object_struct.h"

class YOLOv8RKNN {
private:
    rknn_context ctx = 0;
    rknn_input_output_num io_num;
    std::vector<rknn_tensor_attr> output_attrs;
    std::vector<rknn_output> output_buffers;
    
    // Memory pools for proposals
    mutable std::vector<std::vector<float>> proposals_pool;
    mutable std::vector<int> indices_pool;
    
    static constexpr int STRIDES[3] = {8, 16, 32};

public:
    YOLOv8RKNN() = default;
    YOLOv8RKNN(const YOLOv8RKNN&) = delete;
    YOLOv8RKNN& operator=(const YOLOv8RKNN&) = delete;
    
    // Move constructor and assignment
    YOLOv8RKNN(YOLOv8RKNN&& other) noexcept;
    YOLOv8RKNN& operator=(YOLOv8RKNN&& other) noexcept;
    
    ~YOLOv8RKNN();
    
    bool init(const char* model_path);
    bool infer(const cv::Mat& img, std::vector<Object>& objects,
               float conf_thresh = 0.3f, float nms_thresh = 0.45f) const;
    
private:
    float sigmoid(float x) const;
    float fast_exp(float x) const;
};

#endif // YOLO_DETECTOR_H
