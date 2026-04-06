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
    
    mutable std::vector<std::vector<float>> proposals_pool;
    mutable std::vector<int> indices_pool;
    
    static constexpr int STRIDES[3] = {8, 16, 32};
    
    float nms_sigma_ = 0.5f;
    float iou_threshold_ = 0.45f;
    float multi_label_threshold_ = 0.15f;

public:
    YOLOv8RKNN() = default;
    YOLOv8RKNN(const YOLOv8RKNN&) = delete;
    YOLOv8RKNN& operator=(const YOLOv8RKNN&) = delete;
    
    YOLOv8RKNN(YOLOv8RKNN&& other) noexcept;
    YOLOv8RKNN& operator=(YOLOv8RKNN&& other) noexcept;
    
    ~YOLOv8RKNN();
    
    bool init(const char* model_path);
    
    void set_nms_params(float sigma, float iou_threshold) {
        nms_sigma_ = sigma;
        iou_threshold_ = iou_threshold;
    }
    
    void set_multi_label_threshold(float threshold) {
        multi_label_threshold_ = threshold;
    }
    
    bool infer(const cv::Mat& img, std::vector<Object>& objects,
               float conf_thresh = 0.3f, float nms_thresh = 0.45f) const;
    
private:
    float sigmoid(float x) const;
    float fast_exp(float x) const;
    
    float calculate_iou(const Object& a, const Object& b) const;
    float calculate_diou(const Object& a, const Object& b) const;
    
    void soft_nms(std::vector<Object>& objects) const;
    void apply_multi_label_filtering(std::vector<Object>& objects) const;
    void apply_bbox_coexistence(std::vector<Object>& objects) const;
};

#endif // YOLO_DETECTOR_H
