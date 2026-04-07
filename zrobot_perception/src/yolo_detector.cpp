#include "yolo_detector.h"
#include "colors.h"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cmath>

YOLOv8RKNN::YOLOv8RKNN(YOLOv8RKNN&& other) noexcept
    : ctx(other.ctx)
    , io_num(other.io_num)
    , output_attrs(std::move(other.output_attrs))
    , output_buffers(std::move(other.output_buffers))
    , proposals_pool(std::move(other.proposals_pool))
    , indices_pool(std::move(other.indices_pool))
{
    other.ctx = 0;
}

YOLOv8RKNN& YOLOv8RKNN::operator=(YOLOv8RKNN&& other) noexcept {
    if (this != &other) {
        if (ctx) {
            rknn_destroy(ctx);
        }
        ctx = other.ctx;
        io_num = other.io_num;
        output_attrs = std::move(other.output_attrs);
        output_buffers = std::move(other.output_buffers);
        proposals_pool = std::move(other.proposals_pool);
        indices_pool = std::move(other.indices_pool);
        other.ctx = 0;
    }
    return *this;
}

bool YOLOv8RKNN::init(const char* model_path) {
    int ret = rknn_init(&ctx, (void*)model_path, 0, 0, NULL);
    if (ret < 0) {
        std::cerr << RED << "[YOLO] Failed to init model: " << model_path << RESET << std::endl;
        return false;
    }

    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

    std::cout << GREEN << "[YOLO] Model outputs: " << io_num.n_output << RESET << std::endl;

    output_attrs.resize(io_num.n_output);
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
        
        std::cout << GREEN << "[YOLO] Output " << i 
                  << ": n_dims=" << output_attrs[i].n_dims;
        for (uint32_t d = 0; d < output_attrs[i].n_dims; d++) {
            std::cout << " [" << d << "]=" << output_attrs[i].dims[d];
        }
        std::cout << " type=" << output_attrs[i].type;
        std::cout << RESET << std::endl;
    }

    rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2);
    
    std::cout << GREEN << "[YOLO] Model initialized: " << model_path << RESET << std::endl;
    return true;
}

float YOLOv8RKNN::fast_exp(float x) const {
    union { unsigned int i; float f; } v;
    v.i = (unsigned int)(12102203.161561485f * x + 1065353096.5f);
    return v.f;
}

float YOLOv8RKNN::sigmoid(float x) const {
    return 1.0f / (1.0f + fast_exp(-x));
}

float YOLOv8RKNN::calculate_iou(const Object& a, const Object& b) const {
    float x1 = std::max(a.rect.x, b.rect.x);
    float y1 = std::max(a.rect.y, b.rect.y);
    float x2 = std::min(a.rect.x + a.rect.width, b.rect.x + b.rect.width);
    float y2 = std::min(a.rect.y + a.rect.height, b.rect.y + b.rect.height);

    if (x2 < x1 || y2 < y1) return 0.0f;

    float intersection = (x2 - x1) * (y2 - y1);
    float union_area = a.rect.width * a.rect.height + b.rect.width * b.rect.height - intersection;
    
    return union_area > 0 ? intersection / union_area : 0.0f;
}

float YOLOv8RKNN::calculate_diou(const Object& a, const Object& b) const {
    float iou = calculate_iou(a, b);
    
    float cx1 = a.cx(), cy1 = a.cy();
    float cx2 = b.cx(), cy2 = b.cy();
    
    float center_dist_sq = (cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2);
    
    float enclosed_x1 = std::min(a.rect.x, b.rect.x);
    float enclosed_y1 = std::min(a.rect.y, b.rect.y);
    float enclosed_x2 = std::max(a.rect.x + a.rect.width, b.rect.x + b.rect.width);
    float enclosed_y2 = std::max(a.rect.y + a.rect.height, b.rect.y + b.rect.height);
    
    float diag_dist_sq = (enclosed_x2 - enclosed_x1) * (enclosed_x2 - enclosed_x1) + 
                         (enclosed_y2 - enclosed_y1) * (enclosed_y2 - enclosed_y1);
    
    if (diag_dist_sq <= 0) return iou;
    
    float diou = iou - center_dist_sq / diag_dist_sq;
    return std::max(0.0f, std::min(1.0f, diou));
}

void YOLOv8RKNN::soft_nms(std::vector<Object>& objects) const {
    if (objects.size() < 2) return;

    std::sort(objects.begin(), objects.end(),
              [](const Object& a, const Object& b) { return a.prob > b.prob; });

    std::vector<Object> result;
    std::vector<float> scores(objects.size());
    
    for (size_t i = 0; i < objects.size(); i++) {
        scores[i] = objects[i].prob;
    }

    while (!objects.empty()) {
        size_t max_idx = 0;
        float max_score = objects[0].prob;
        
        for (size_t i = 1; i < objects.size(); i++) {
            if (objects[i].prob > max_score) {
                max_score = objects[i].prob;
                max_idx = i;
            }
        }

        result.push_back(objects[max_idx]);
        objects.erase(objects.begin() + max_idx);
        scores.erase(scores.begin() + max_idx);

        if (objects.empty()) break;

        for (size_t i = 0; i < objects.size(); i++) {
            float diou = calculate_diou(result.back(), objects[i]);
            
            if (diou > iou_threshold_) {
                float decay = std::exp(-(diou * diou) / nms_sigma_);
                objects[i].prob *= decay;
                scores[i] *= decay;
            }
        }

        objects.erase(
            std::remove_if(objects.begin(), objects.end(),
                [](const Object& obj) { return obj.prob < 0.001f; }),
            objects.end()
        );
        
        scores.erase(
            std::remove_if(scores.begin(), scores.end(),
                [](float s) { return s < 0.001f; }),
            scores.end()
        );
    }

    objects = result;
}

void YOLOv8RKNN::apply_multi_label_filtering(std::vector<Object>& objects) const {
    for (auto& obj : objects) {
        obj.multi_labels.clear();
        obj.multi_labels.emplace_back(obj.label, obj.prob);
    }
}

void YOLOv8RKNN::apply_bbox_coexistence(std::vector<Object>& objects) const {
    if (objects.size() < 2) return;

    std::vector<bool> suppressed(objects.size(), false);
    
    for (size_t i = 0; i < objects.size(); i++) {
        if (suppressed[i]) continue;
        
        for (size_t j = i + 1; j < objects.size(); j++) {
            if (suppressed[j]) continue;

            float center_dist = std::sqrt(
                std::pow(objects[i].cx() - objects[j].cx(), 2) +
                std::pow(objects[i].cy() - objects[j].cy(), 2)
            );

            float avg_width = (objects[i].rect.width + objects[j].rect.width) * 0.5f;
            
            if (center_dist < avg_width * 0.3f) {
                if (objects[i].prob > objects[j].prob) {
                    suppressed[j] = true;
                } else if (objects[j].prob > objects[i].prob) {
                    suppressed[j] = true;
                } else {
                    float iou = calculate_iou(objects[i], objects[j]);
                    if (iou > 0.5f) {
                        if (objects[i].rect.area() > objects[j].rect.area()) {
                            suppressed[j] = true;
                        } else {
                            suppressed[i] = true;
                            break;
                        }
                    }
                }
            }
        }
    }

    std::vector<Object> filtered;
    for (size_t i = 0; i < objects.size(); i++) {
        if (!suppressed[i]) {
            filtered.push_back(objects[i]);
        }
    }
    objects = filtered;
}

inline float dequantize_int8(int8_t val, float scale, int32_t zero_point) {
    return (float)(val - zero_point) * scale;
}

bool YOLOv8RKNN::infer(const cv::Mat& img, std::vector<Object>& objects,
                       float conf_thresh, float nms_thresh) const {
    (void)nms_thresh;
    
    int input_w = 640, input_h = 640;
    float scale = std::min((float)input_w / img.cols, (float)input_h / img.rows);
    int new_w = img.cols * scale;
    int new_h = img.rows * scale;
    int pad_x = (input_w - new_w) / 2;
    int pad_y = (input_h - new_h) / 2;

    cv::Mat padded(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));
    resized.copyTo(padded(cv::Rect(pad_x, pad_y, new_w, new_h)));

    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    std::vector<rknn_output> outputs(io_num.n_output);
    for (auto& out : outputs) {
        out.want_float = 1;
        out.is_prealloc = 0;
    }

    rknn_input input;
    memset(&input, 0, sizeof(input));
    input.index = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.size = rgb.total() * 3;
    input.fmt = RKNN_TENSOR_NHWC;
    input.buf = rgb.data;

    rknn_inputs_set(ctx, 1, &input);
    rknn_run(ctx, nullptr);
    rknn_outputs_get(ctx, io_num.n_output, outputs.data(), nullptr);

    objects.clear();
    std::vector<std::pair<int, float>> cls_probs(80);

    if (io_num.n_output >= 1) {
        int8_t* output_int8 = (int8_t*)outputs[0].buf;
        
        float output_scale = output_attrs[0].scale;
        int32_t output_zero_point = 0;
        
        int num_classes = output_attrs[0].dims[1];
        int num_anchors = output_attrs[0].dims[2];
        
        int strides[3] = {8, 16, 32};
        int grid_sizes[3] = {80, 40, 20};
        
        int offset = 0;
        
        for (int scale_idx = 0; scale_idx < 3; scale_idx++) {
            int grid_h = grid_sizes[scale_idx];
            int grid_w = grid_sizes[scale_idx];
            int num_anchors_scale = grid_h * grid_w;
            int stride = strides[scale_idx];
            
            for (int h = 0; h < grid_h; h++) {
                for (int w = 0; w < grid_w; w++) {
                    int idx = offset + h * grid_w + w;
                    
                    float tx = dequantize_int8(output_int8[idx], output_scale, output_zero_point);
                    float ty = dequantize_int8(output_int8[num_anchors + idx], output_scale, output_zero_point);
                    float tw = dequantize_int8(output_int8[num_anchors * 2 + idx], output_scale, output_zero_point);
                    float th = dequantize_int8(output_int8[num_anchors * 3 + idx], output_scale, output_zero_point);
                    
                    float obj_score = sigmoid(dequantize_int8(output_int8[num_anchors * 4 + idx], output_scale, output_zero_point));
                    if (obj_score < 0.001f) continue;
                    
                    float max_cls = 0.0f;
                    int cls_id = -1;
                    
                    for (int c = 0; c < num_classes; c++) {
                        float cls_score = sigmoid(dequantize_int8(output_int8[num_anchors * (4 + 1 + c) + idx], output_scale, output_zero_point));
                        cls_probs[c] = {c, cls_score};
                        if (cls_score > max_cls) {
                            max_cls = cls_score;
                            cls_id = c;
                        }
                    }
                    
                    float final_conf = obj_score * max_cls;
                    if (final_conf < conf_thresh || cls_id < 0) continue;
                    
                    float bx = (sigmoid(tx) * 2 - 0.5f + w) * stride;
                    float by = (sigmoid(ty) * 2 - 0.5f + h) * stride;
                    float bw = (sigmoid(tw) * 2) * stride;
                    float bh = (sigmoid(th) * 2) * stride;
                    
                    float x1 = bx - bw / 2.0f;
                    float y1 = by - bh / 2.0f;
                    float x2 = bx + bw / 2.0f;
                    float y2 = by + bh / 2.0f;
                    
                    x1 = (x1 - pad_x) / scale;
                    y1 = (y1 - pad_y) / scale;
                    x2 = (x2 - pad_x) / scale;
                    y2 = (y2 - pad_y) / scale;
                    
                    x1 = std::max(0.0f, std::min(x1, (float)img.cols));
                    y1 = std::max(0.0f, std::min(y1, (float)img.rows));
                    x2 = std::max(0.0f, std::min(x2, (float)img.cols));
                    y2 = std::max(0.0f, std::min(y2, (float)img.rows));
                    
                    if (x2 > x1 && y2 > y1) {
                        Object obj;
                        obj.rect = cv::Rect_<float>(x1, y1, x2-x1, y2-y1);
                        obj.label = cls_id;
                        obj.prob = final_conf;
                        
                        for (const auto& p : cls_probs) {
                            if (p.second > multi_label_threshold_ && p.second > max_cls * 0.5f) {
                                obj.multi_labels.emplace_back(p.first, p.second * obj_score);
                            }
                        }
                        
                        objects.push_back(obj);
                    }
                }
            }
            
            offset += num_anchors_scale;
        }
    }

    rknn_outputs_release(ctx, io_num.n_output, outputs.data());

    if (!objects.empty()) {
        std::sort(objects.begin(), objects.end(),
                  [](const Object& a, const Object& b) { return a.prob > b.prob; });

        soft_nms(objects);
        
        apply_multi_label_filtering(objects);
        
        apply_bbox_coexistence(objects);
    }

    return true;
}

YOLOv8RKNN::~YOLOv8RKNN() {
    if (ctx) {
        rknn_destroy(ctx);
    }
}
