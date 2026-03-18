#include "yolo_detector.h"
#include "colors.h"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cmath>

// Move constructor
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

// Move assignment
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

    output_attrs.resize(io_num.n_output);
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
    }

    // Use all NPU cores on RK3588
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

bool YOLOv8RKNN::infer(const cv::Mat& img, std::vector<Object>& objects,
                       float conf_thresh, float nms_thresh) const {
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

    // Process 3 detection heads
    for (int s = 0; s < 3; s++) {
        float* bbox_data = (float*)outputs[s*3 + 0].buf;
        float* cls_data  = (float*)outputs[s*3 + 1].buf;
        float* obj_data  = (float*)outputs[s*3 + 2].buf;

        int H = output_attrs[s*3 + 1].dims[2];
        int W = output_attrs[s*3 + 1].dims[3];
        int stride = STRIDES[s];
        int area = H * W;

        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int idx = h * W + w;
                float obj_conf = obj_data[idx];
                if (obj_conf < 0.001f) continue;

                float max_cls = 0.0f;
                int cls_id = -1;
                for (int c = 0; c < 80; c++) {
                    float cls_prob = sigmoid(cls_data[c * area + idx]);
                    if (cls_prob > max_cls) {
                        max_cls = cls_prob;
                        cls_id = c;
                    }
                }

                float final_conf = obj_conf * max_cls;
                if (final_conf < conf_thresh) continue;

                // Decode bounding box
                float box[4] = {0};
                for (int k = 0; k < 4; k++) {
                    float max_val = -FLT_MAX;
                    float exp_vals[16];

                    for (int d = 0; d < 16; d++) {
                        float v = bbox_data[(k * 16 + d) * area + idx];
                        if (v > max_val) max_val = v;
                    }

                    float sum = 0;
                    for (int d = 0; d < 16; d++) {
                        exp_vals[d] = fast_exp(bbox_data[(k * 16 + d) * area + idx] - max_val);
                        sum += exp_vals[d];
                    }

                    box[k] = 0;
                    for (int d = 0; d < 16; d++) {
                        box[k] += d * (exp_vals[d] / sum);
                    }
                }

                float cx = (w + 0.5f) * stride;
                float cy = (h + 0.5f) * stride;
                float x1 = cx - box[0] * stride;
                float y1 = cy - box[1] * stride;
                float x2 = cx + box[2] * stride;
                float y2 = cy + box[3] * stride;

                // Transform to original image coordinates
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
                    objects.push_back(obj);
                }
            }
        }
    }

    rknn_outputs_release(ctx, io_num.n_output, outputs.data());

    // Cluster nearby detections
    if (!objects.empty()) {
        std::sort(objects.begin(), objects.end(),
                 [](const Object& a, const Object& b) { return a.prob > b.prob; });

        std::vector<bool> processed(objects.size(), false);
        std::vector<Object> consolidated_objects;

        for (size_t i = 0; i < objects.size(); i++) {
            if (processed[i]) continue;

            Object cluster_obj = objects[i];
            std::vector<size_t> cluster_members = {i};

            for (size_t j = 0; j < objects.size(); j++) {
                if (i == j || processed[j]) continue;
                if (objects[i].label != objects[j].label) continue;

                float center_dist = std::sqrt(std::pow(
                    objects[i].rect.x + objects[i].rect.width/2.0f -
                    (objects[j].rect.x + objects[j].rect.width/2.0f), 2) +
                    std::pow(objects[i].rect.y + objects[i].rect.height/2.0f -
                    (objects[j].rect.y + objects[j].rect.height/2.0f), 2));

                float combined_width = std::max(objects[i].rect.br().x, objects[j].rect.br().x) -
                                     std::min(objects[i].rect.x, objects[j].rect.x);
                float combined_height = std::max(objects[i].rect.br().y, objects[j].rect.br().y) -
                                      std::min(objects[i].rect.y, objects[j].rect.y);
                float avg_size = (combined_width + combined_height) / 2.0f;

                if (center_dist < avg_size * 2.0f) {
                    float x1 = std::min(cluster_obj.rect.x, objects[j].rect.x);
                    float y1 = std::min(cluster_obj.rect.y, objects[j].rect.y);
                    float x2 = std::max(cluster_obj.rect.br().x, objects[j].rect.br().x);
                    float y2 = std::max(cluster_obj.rect.br().y, objects[j].rect.br().y);

                    cluster_obj.rect = cv::Rect_<float>(x1, y1, x2-x1, y2-y1);
                    float weight_i = objects[i].rect.area();
                    float weight_j = objects[j].rect.area();
                    cluster_obj.prob = (cluster_obj.prob * weight_i + objects[j].prob * weight_j) / 
                                      (weight_i + weight_j);
                    cluster_members.push_back(j);
                    processed[j] = true;
                }
            }

            processed[i] = true;
            consolidated_objects.push_back(cluster_obj);
        }

        objects = consolidated_objects;
    }

    return true;
}

YOLOv8RKNN::~YOLOv8RKNN() {
    if (ctx) {
        rknn_destroy(ctx);
    }
}
