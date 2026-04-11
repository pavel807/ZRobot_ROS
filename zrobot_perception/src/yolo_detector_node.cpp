#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float32.hpp>
#include <cv_bridge/cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <chrono>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <deque>
#include <iomanip>
#include <sstream>

#include "yolo_detector.h"
#include "object_search.h"
#include "colors.h"
#include "object_struct.h"
#include "coco_classes.h"
#include "object_tracker.h"

// ============================================================================
// DETECTION PARAMETERS — синхронизировано с оригиналом ZRobot v6.3
// ============================================================================
struct DetectionParams {
    float conf_threshold = 0.30f;          // Как в оригинале
    float nms_threshold = 0.45f;            // Как в оригинале
    float min_area_ratio = 0.002f;
    float max_area_ratio = 0.45f;
    bool enable_tracking = true;
    bool temporal_filtering = true;
    int history_size = 35;
    float velocity_weight = 0.25f;
    float confidence_decay = 0.95f;
    int max_consecutive_misses = 10;
    float iou_threshold = 0.35f;
    int min_confirm_frames = 2;              // ИСПРАВЛЕНО: было 5 → теперь 2 (быстрая реакция)
    float hysteresis_margin = 0.05f;
    int class_stability_window = 3;          // ИСПРАВЛЕНО: было 5 → теперь 3
    float position_smoothing = 0.6f;
};

// ============================================================================
// MOTOR CONTROL PARAMETERS — полностью синхронизировано с оригиналом ZRobot
// ============================================================================
struct MotorControlParams {
    std::atomic<int> max_speed{245};                     // Как в оригинале
    std::atomic<int> min_speed{165};                     // Как в оригинале
    std::atomic<int> tracking_speed{220};                // Как в оригинале
    std::atomic<int> search_speed{115};                  // Как в оригинале
    std::atomic<float> center_threshold{0.08f};          // Как в оригинале
    std::atomic<float> max_turn_ratio{0.95f};            // Как в оригинале
    std::atomic<int> approach_width{120};                // Как в оригинале
    std::atomic<int> search_timeout_ms{300};             // Как в оригинале
    std::atomic<int> search_hysteresis_ms{400};          // Как в оригинале
    std::atomic<int> speed_smoothing_steps{4};           // Как в оригинале
};

// ============================================================================
// PERFORMANCE MONITOR
// ============================================================================
class PerformanceMonitor {
public:
    void recordInference(float ms) {
        inference_times_.push_back(ms);
        if (inference_times_.size() > 60) inference_times_.pop_front();
    }

    void recordFPS(float fps) {
        fps_history_.push_back(fps);
        if (fps_history_.size() > 60) fps_history_.pop_front();
    }

    float getAverageInference() const {
        if (inference_times_.empty()) return 0.0f;
        float sum = 0.0f;
        for (float t : inference_times_) sum += t;
        return sum / inference_times_.size();
    }

    float getAverageFPS() const {
        if (fps_history_.empty()) return 0.0f;
        float sum = 0.0f;
        for (float f : fps_history_) sum += f;
        return sum / fps_history_.size();
    }

    float getMinInference() const {
        return *std::min_element(inference_times_.begin(), inference_times_.end());
    }

    float getMaxInference() const {
        return *std::max_element(inference_times_.begin(), inference_times_.end());
    }

    std::string getStats() const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1);
        oss << "FPS: " << getAverageFPS()
            << " | Inf: " << getAverageInference() << "ms"
            << " [" << getMinInference() << "-" << getMaxInference() << "]";
        return oss.str();
    }

private:
    std::deque<float> inference_times_;
    std::deque<float> fps_history_;
};

// ============================================================================
// ADAPTIVE THRESHOLD CONTROLLER
// ============================================================================
class AdaptiveThresholdController {
public:
    void update(float current_fps, float target_fps = 25.0f) {
        if (current_fps < target_fps * 0.8f) {
            conf_threshold_ = std::min(0.75f, conf_threshold_ + 0.02f);
        } else if (current_fps > target_fps * 1.2f) {
            conf_threshold_ = std::max(0.25f, conf_threshold_ - 0.01f);
        }
    }

    float getThreshold() const { return conf_threshold_; }
    void reset(float value) { conf_threshold_ = value; }

private:
    float conf_threshold_ = 0.30f;          // Синхронизировано с оригиналом
};

// ============================================================================
// DETECTION STABILITY CONTROLLER — ускоренная реакция
// ============================================================================
class DetectionStabilityController {
public:
    DetectionStabilityController() : last_detection_state_(false), stable_count_(0), unstable_count_(0) {}

    bool isStableDetection(bool current_detected, float confidence) {
        if (current_detected && confidence >= 0.40f) {
            stable_count_++;
            unstable_count_ = 0;
            if (stable_count_ >= 2) {           // ИСПРАВЛЕНО: было 3 → теперь 2
                last_detection_state_ = true;
                return true;
            }
        } else if (!current_detected) {
            unstable_count_++;
            stable_count_ = 0;
            if (unstable_count_ >= 3) {         // ИСПРАВЛЕНО: было 4 → теперь 3
                last_detection_state_ = false;
            }
        } else if (confidence >= 0.30f && confidence < 0.40f) {
            stable_count_++;
            unstable_count_ = 0;
            if (stable_count_ >= 3) {           // ИСПРАВЛЕНО: было 5 → теперь 3
                last_detection_state_ = true;
                return true;
            }
        }
        return last_detection_state_;
    }

    void reset() {
        stable_count_ = 0;
        unstable_count_ = 0;
        last_detection_state_ = false;
    }

private:
    bool last_detection_state_;
    int stable_count_;
    int unstable_count_;
};

// ============================================================================
// CLASS STABILITY VERIFIER
// ============================================================================
class ClassStabilityVerifier {
public:
    ClassStabilityVerifier(int window_size = 3) : window_size_(window_size) {}

    std::string getStableClass(const std::string& detected_class, float confidence) {
        if (confidence < 0.40f) {
            return last_stable_class_;
        }

        class_history_.push_back(detected_class);
        if (class_history_.size() > window_size_) {
            class_history_.pop_front();
        }

        std::unordered_map<std::string, int> class_counts;
        for (const auto& c : class_history_) {
            class_counts[c]++;
        }

        int max_count = 0;
        std::string most_common;
        for (const auto& [cls, cnt] : class_counts) {
            if (cnt > max_count) {
                max_count = cnt;
                most_common = cls;
            }
        }

        if (max_count >= 2) {                   // ИСПРАВЛЕНО: было 3 → теперь 2
            last_stable_class_ = most_common;
        }

        return last_stable_class_;
    }

    void reset() {
        class_history_.clear();
        last_stable_class_ = "";
    }

private:
    std::deque<std::string> class_history_;
    std::string last_stable_class_;
    size_t window_size_;
};

// ============================================================================
// POSITION SMOOTHER
// ============================================================================
class PositionSmoother {
public:
    PositionSmoother(float smoothing_factor = 0.6f) : smoothing_factor_(smoothing_factor), initialized_(false) {}

    cv::Point2f smooth(const cv::Point2f& new_position) {
        if (!initialized_) {
            smoothed_position_ = new_position;
            initialized_ = true;
            return new_position;
        }

        smoothed_position_.x = smoothing_factor_ * new_position.x + (1.0f - smoothing_factor_) * smoothed_position_.x;
        smoothed_position_.y = smoothing_factor_ * new_position.y + (1.0f - smoothing_factor_) * smoothed_position_.y;
        return smoothed_position_;
    }

    void reset() {
        initialized_ = false;
    }

private:
    cv::Point2f smoothed_position_;
    float smoothing_factor_;
    bool initialized_;
};

// ============================================================================
// YOLO DETECTOR NODE — ПОЛНОСТЬЮ ПЕРЕПИСАН С ПОИСКОМ, ГИСТЕРЕЗИСОМ, СГЛАЖИВАНИЕМ
// ============================================================================
class YoloDetectorNode : public rclcpp::Node {
public:
    YoloDetectorNode() : Node("yolo_detector") {
        RCLCPP_INFO(this->get_logger(), "Initializing Advanced YOLO Detector...");
        RCLCPP_INFO(this->get_logger(), "COCO Classes: %zu objects in %zu categories",
                    COCO_CLASSES.size(), COCO_CATEGORIES.size());

        // Declare parameters
        this->declare_parameter<std::string>("model_path", "models/yolo26s-rk3588.rknn");
        this->declare_parameter<int>("camera_id", 0);
        this->declare_parameter<int>("width", 640);
        this->declare_parameter<int>("height", 640);
        this->declare_parameter<int>("fps", 30);
        this->declare_parameter<float>("conf_threshold", 0.30f);
        this->declare_parameter<std::string>("target_object", "person");
        this->declare_parameter<bool>("enable_tracking", true);
        this->declare_parameter<bool>("adaptive_threshold", false);
        this->declare_parameter<bool>("show_category", true);
        this->declare_parameter<bool>("enable_auto_follow", true);
        this->declare_parameter<double>("max_linear_speed", 0.3);
        this->declare_parameter<double>("turn_speed", 0.5);
        this->declare_parameter<double>("nms_sigma", 0.5);
        this->declare_parameter<double>("iou_threshold", 0.45);
        this->declare_parameter<double>("multi_label_threshold", 0.15);

        // Get parameters
        std::string model_path;
        int camera_id, width, height, fps;
        float conf_threshold;
        bool enable_tracking, adaptive_threshold, show_category;
        bool enable_auto_follow;
        double max_linear_speed, turn_speed;
        double nms_sigma, iou_threshold, multi_label_threshold;

        this->get_parameter("model_path", model_path);
        this->get_parameter("camera_id", camera_id);
        this->get_parameter("width", width);
        this->get_parameter("height", height);
        this->get_parameter("fps", fps);
        this->get_parameter("conf_threshold", conf_threshold);
        this->get_parameter("target_object", target_object_);
        this->get_parameter("enable_tracking", enable_tracking);
        this->get_parameter("adaptive_threshold", adaptive_threshold);
        this->get_parameter("show_category", show_category);
        this->get_parameter("enable_auto_follow", enable_auto_follow);
        this->get_parameter("max_linear_speed", max_linear_speed);
        this->get_parameter("turn_speed", turn_speed);
        this->get_parameter("nms_sigma", nms_sigma);
        this->get_parameter("iou_threshold", iou_threshold);
        this->get_parameter("multi_label_threshold", multi_label_threshold);

        params_.conf_threshold = conf_threshold;
        params_.enable_tracking = enable_tracking;
        adaptive_threshold_ = adaptive_threshold;
        show_category_ = show_category;
        enable_auto_follow_ = enable_auto_follow;
        max_linear_speed_ = max_linear_speed;
        turn_speed_ = turn_speed;

        yolo_.set_nms_params(nms_sigma, iou_threshold);
        yolo_.set_multi_label_threshold(multi_label_threshold);

        // Initialize YOLO detector
        if (!yolo_.init(model_path.c_str())) {
            RCLCPP_ERROR(this->get_logger(), "[YoloDetector] Failed to initialize YOLO model!");
            exit(1);
        }

        // Initialize camera with retry
        int retry = 3;
        while (retry > 0) {
            cap_.open(camera_id, cv::CAP_V4L2);
            if (cap_.isOpened()) break;
            RCLCPP_WARN(this->get_logger(), "Camera %d not found, retrying... (%d)", camera_id, retry);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            retry--;
        }

        if (!cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "[YoloDetector] Failed to open camera %d", camera_id);
            exit(1);
        }

        cap_.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        cap_.set(cv::CAP_PROP_FPS, fps);
        cap_.set(cv::CAP_PROP_AUTOFOCUS, 1);
        cap_.set(cv::CAP_PROP_AUTO_EXPOSURE, 3);

        int actual_width = cap_.get(cv::CAP_PROP_FRAME_WIDTH);
        int actual_height = cap_.get(cv::CAP_PROP_FRAME_HEIGHT);
        float actual_fps = cap_.get(cv::CAP_PROP_FPS);

        RCLCPP_INFO(this->get_logger(), "Camera initialized: %dx%d @ %.1f FPS",
                    actual_width, actual_height, actual_fps);

        // Create publishers
        detections_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("detections", 10);
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("processed_image", 10);
        status_pub_ = this->create_publisher<std_msgs::msg::String>("detection_status", 10);
        tracked_pub_ = this->create_publisher<std_msgs::msg::String>("tracked_objects", 10);
        cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);

        // Create subscription for target object changes
        target_sub_ = this->create_subscription<std_msgs::msg::String>(
            "set_target", 10,
            std::bind(&YoloDetectorNode::targetCallback, this, std::placeholders::_1)
        );

        // Create subscription for confidence threshold changes
        confidence_sub_ = this->create_subscription<std_msgs::msg::Float32>(
            "set_confidence", 10,
            std::bind(&YoloDetectorNode::confidenceCallback, this, std::placeholders::_1)
        );

        // Create timer for detection loop
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000 / fps),
            std::bind(&YoloDetectorNode::detectLoop, this)
        );

        frame_count_ = 0;
        last_fps_time_ = this->now();
        current_fps_ = 0.0;
        base_conf_threshold_ = conf_threshold;

        // Инициализация состояния поиска/слежения
        last_detection_time_ = std::chrono::steady_clock::now();
        search_start_time_ = std::chrono::steady_clock::now();
        was_target_found_ = false;
        in_search_mode_ = false;
        is_searching_ = false;
        search_direction_ = 1;
        has_been_detected_ = false;
        last_known_zone_ = "NONE";

        // Инициализация истории скоростей для сглаживания
        speed_history_l_.clear();
        speed_history_r_.clear();

        RCLCPP_INFO(this->get_logger(), "Advanced YOLO Detector node started");
        RCLCPP_INFO(this->get_logger(), "Target: %s | Tracking: %s | Adaptive: %s",
                    target_object_.c_str(),
                    enable_tracking ? "ON" : "OFF",
                    adaptive_threshold ? "ON" : "OFF");
        RCLCPP_INFO(this->get_logger(), "Motor params: max=%d min=%d track=%d search=%d",
                    motor_params_.max_speed.load(), motor_params_.min_speed.load(),
                    motor_params_.tracking_speed.load(), motor_params_.search_speed.load());
    }

private:
    void targetCallback(const std_msgs::msg::String::SharedPtr msg) {
        if (target_object_ != msg->data) {
            target_object_ = msg->data;
            RCLCPP_INFO(this->get_logger(), "Target changed to: %s - resetting stability", target_object_.c_str());
            if (params_.enable_tracking) {
                tracker_.clear();
            }
            stability_controller_.reset();
            class_verifier_.reset();
            position_smoothers_.clear();

            // Сброс состояния поиска
            in_search_mode_ = false;
            is_searching_ = false;
            has_been_detected_ = false;
            last_known_zone_ = "NONE";
            search_direction_ = 1;
            speed_history_l_.clear();
            speed_history_r_.clear();
        }
    }

    void confidenceCallback(const std_msgs::msg::Float32::SharedPtr msg) {
        float new_confidence = msg->data;
        if (new_confidence >= 0.0f && new_confidence <= 1.0f) {
            float old_confidence = params_.conf_threshold;
            params_.conf_threshold = new_confidence;
            base_conf_threshold_ = new_confidence;
            RCLCPP_INFO(this->get_logger(), "Confidence threshold changed: %.2f -> %.2f", 
                        old_confidence, new_confidence);
        } else {
            RCLCPP_WARN(this->get_logger(), "Invalid confidence value: %.2f (must be 0.0-1.0)", new_confidence);
        }
    }

    void detectLoop() {
        cv::Mat frame;
        cap_ >> frame;

        if (frame.empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Empty frame received");
            return;
        }

        // Start inference timing
        auto t1 = std::chrono::high_resolution_clock::now();

        // Run YOLO inference
        std::vector<Object> raw_objects;
        yolo_.infer(frame, raw_objects, params_.conf_threshold, params_.nms_threshold);

        auto t2 = std::chrono::high_resolution_clock::now();
        float inference_time_ms = std::chrono::duration<float, std::milli>(t2 - t1).count();
        perf_monitor_.recordInference(inference_time_ms);

        // Filter by area
        float frame_area = frame.cols * frame.rows;
        std::vector<Object> filtered_objects;
        for (const auto& obj : raw_objects) {
            float area_ratio = obj.rect.area() / frame_area;
            if (area_ratio >= params_.min_area_ratio && area_ratio <= params_.max_area_ratio) {
                filtered_objects.push_back(obj);
            }
        }

        // Apply tracking
        std::vector<TrackedObject> tracked_objects;
        if (params_.enable_tracking && !filtered_objects.empty()) {
            std::vector<cv::Rect> rects;
            std::vector<int> labels;
            std::vector<std::string> names;
            std::vector<float> confs;

            for (const auto& obj : filtered_objects) {
                rects.push_back(obj.rect);
                labels.push_back(obj.label);
                names.push_back(obj.label >= 0 && obj.label < 80 ? COCO_NAMES[obj.label] : "unknown");
                confs.push_back(obj.prob);
            }

            tracked_objects = tracker_.update(rects, labels, names, confs);
        }

        // Update FPS
        frame_count_++;
        auto now = this->now();
        double elapsed = (now - last_fps_time_).seconds();
        if (elapsed >= 0.5) {
            current_fps_ = frame_count_ / elapsed;
            perf_monitor_.recordFPS(current_fps_);

            if (adaptive_threshold_) {
                threshold_controller_.update(current_fps_);
            }

            frame_count_ = 0;
            last_fps_time_ = now;
        }

        // Publish detections + auto-follow control
        publishDetections(filtered_objects, tracked_objects, frame, inference_time_ms);

        // Update FPS counter on image with stability status
        drawOverlay(frame, last_target_stable_);

        // Publish processed image
        if (image_pub_->get_subscription_count() > 0) {
            std_msgs::msg::Header header;
            header.stamp = this->now();
            header.frame_id = "camera";
            auto img_msg = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, frame).toImageMsg();
            image_pub_->publish(*img_msg);
        }
    }

    void drawOverlay(cv::Mat& frame, bool target_is_stable) {
        std::string stats = perf_monitor_.getStats();
        cv::putText(frame, stats, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        char target_text[64];
        snprintf(target_text, sizeof(target_text), "Target: %s", target_object_.c_str());
        cv::putText(frame, target_text, cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

        char thresh_text[64];
        snprintf(thresh_text, sizeof(thresh_text), "Conf: %.2f%s",
                 adaptive_threshold_ ? threshold_controller_.getThreshold() : params_.conf_threshold,
                 adaptive_threshold_ ? " (adaptive)" : "");
        cv::putText(frame, thresh_text, cv::Point(10, 85),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 200, 0), 2);

        if (params_.enable_tracking) {
            char track_text[64];
            snprintf(track_text, sizeof(track_text), "Tracked: %zu | Stable: %s",
                     tracker_.getActiveTrackCount(),
                     target_is_stable ? "YES" : "NO");
            cv::putText(frame, track_text, cv::Point(10, 110),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 255), 2);
        }

        // Режим: TRACKING / SEARCHING
        std::string mode_status = in_search_mode_ ? "SEARCHING" : (target_is_stable ? "TRACKING" : "IDLE");
        cv::Scalar mode_color = in_search_mode_ ? cv::Scalar(0, 165, 255) :
                                (target_is_stable ? cv::Scalar(0, 255, 0) : cv::Scalar(128, 128, 128));
        cv::putText(frame, mode_status, cv::Point(frame.cols - 140, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2);
    }

    // =========================================================================
    //calculateApproachSpeed() — прямо как в оригинале ZRobot
    // =========================================================================
    int calculateApproachSpeed(int obj_width) {
        int base_speed = motor_params_.tracking_speed.load();

        if (obj_width > motor_params_.approach_width.load()) {
            float distance_factor = (float)(obj_width - motor_params_.approach_width.load()) / 300.0f;
            distance_factor = std::min(1.0f, distance_factor);
            float scale = 1.0f - (distance_factor * 0.7f);
            scale = std::max(0.25f, scale);
            base_speed = (int)(base_speed * scale);
        }

        return std::max(motor_params_.min_speed.load(), base_speed);
    }

    // =========================================================================
    // smoothSpeeds() — скользящее среднее как в оригинале
    // =========================================================================
    void smoothSpeeds(int target_left, int target_right, int& out_left, int& out_right) {
        speed_history_l_.push_back(target_left);
        speed_history_r_.push_back(target_right);

        int steps = motor_params_.speed_smoothing_steps.load();
        while ((int)speed_history_l_.size() > steps) {
            speed_history_l_.pop_front();
        }
        while ((int)speed_history_r_.size() > steps) {
            speed_history_r_.pop_front();
        }

        int sum_l = 0, sum_r = 0;
        for (int s : speed_history_l_) sum_l += s;
        for (int s : speed_history_r_) sum_r += s;

        out_left = sum_l / (int)speed_history_l_.size();
        out_right = sum_r / (int)speed_history_r_.size();
    }

    // =========================================================================
    // calculateMotorSpeedsProportional() — ПРЯМО КАК В ОРИГИНАЛЕ ZRobot
    // =========================================================================
    void calculateMotorSpeedsProportional(float center_x, int frame_width, bool found, int obj_width,
                                          int& out_left, int& out_right) {
        out_left = 0;
        out_right = 0;

        // ЛОГИКА ПОИСКА
        if (!found || frame_width <= 0) {
            if (is_searching_) {
                int search_dir = search_direction_;
                int search_speed = motor_params_.search_speed.load();

                if (search_dir > 0) {
                    // Вправо: левое вперед, правое стоп
                    out_left = search_speed;
                    out_right = 0;
                } else {
                    // Влево: правое вперед, левое стоп
                    out_left = 0;
                    out_right = search_speed;
                }
            } else {
                out_left = 0;
                out_right = 0;
            }
            return;
        }

        // ОБЫЧНАЯ ЛОГИКА СЛЕЖЕНИЯ
        float frame_center = frame_width / 2.0f;
        float deviation = (center_x - frame_center) / frame_center;
        int base_speed = calculateApproachSpeed(obj_width);
        float abs_deviation = fabs(deviation);

        if (abs_deviation < motor_params_.center_threshold.load()) {
            out_left = base_speed;
            out_right = base_speed;
        } else {
            float turn_intensity = (abs_deviation - motor_params_.center_threshold.load()) /
                                   (1.0f - motor_params_.center_threshold.load());
            turn_intensity = std::min(1.0f, turn_intensity);
            float speed_ratio = 1.0f - (turn_intensity * motor_params_.max_turn_ratio.load());

            if (deviation > 0) {
                out_left = base_speed;
                out_right = (int)(base_speed * speed_ratio);
            } else {
                out_left = (int)(base_speed * speed_ratio);
                out_right = base_speed;
            }
        }

        out_left = std::max(0, std::min(255, out_left));
        out_right = std::max(0, std::min(255, out_right));
    }

    // =========================================================================
    // convertPWMtoTwist — конвертация PWM (0-255) → нормализованный Twist (0.0-1.0)
    // =========================================================================
    void convertPWMtoTwist(int pwm_left, int pwm_right, double& out_linear, double& out_angular) {
        // Дифференциальная кинематика: обратное преобразование
        float linear_f = (pwm_left + pwm_right) / 2.0f;
        float angular_f = (pwm_right - pwm_left) / 2.0f;

        // Нормализация относительно max_speed
        float max_spd = (float)motor_params_.max_speed.load();
        out_linear = (double)(linear_f / max_spd);
        out_angular = (double)(angular_f / max_spd);

        // Clamp
        out_linear = std::max(-1.0, std::min(1.0, out_linear));
        out_angular = std::max(-1.0, std::min(1.0, out_angular));
    }

    void publishDetections(const std::vector<Object>& objects,
                           const std::vector<TrackedObject>& tracked_objects,
                           const cv::Mat& frame, float inference_time_ms) {
        if (detections_pub_->get_subscription_count() == 0 &&
            status_pub_->get_subscription_count() == 0) {
            return;
        }

        vision_msgs::msg::Detection2DArray detection_array;
        detection_array.header.stamp = this->now();
        detection_array.header.frame_id = "camera";

        bool target_found = false;
        std::string target_zone = "NONE";
        float target_center_x = 0.0f;
        float target_confidence = 0.0f;  // Confidence целевого объекта
        int target_count = 0;
        cv::Rect best_target_rect;
        float best_target_area = 0.0f;
        int best_target_width = 0;
        std::string detected_classes;

        const auto& active_objects = tracked_objects;

        for (size_t i = 0; i < active_objects.size(); i++) {
            const auto& obj = active_objects[i];
            std::string class_name = obj.class_name;

            float avg_conf = obj.getAverageConfidence();
            class_name = class_verifier_.getStableClass(class_name, avg_conf);

            cv::Rect rect = obj.history.back();
            cv::Point2f center(rect.x + rect.width/2.0f, rect.y + rect.height/2.0f);

            if (position_smoothers_.find(obj.id) == position_smoothers_.end()) {
                position_smoothers_[obj.id] = PositionSmoother(params_.position_smoothing);
            }
            cv::Point2f smoothed_center = position_smoothers_[obj.id].smooth(center);

            rect.x = static_cast<int>(smoothed_center.x - rect.width/2.0f);
            rect.y = static_cast<int>(smoothed_center.y - rect.height/2.0f);
            rect = rect & cv::Rect(0, 0, frame.cols, frame.rows);

            auto cat_it = ID_TO_CATEGORY.find(obj.label);
            std::string category = (cat_it != ID_TO_CATEGORY.end()) ? cat_it->second : "unknown";

            if (class_name == target_object_) {
                bool is_stable = stability_controller_.isStableDetection(true, avg_conf);
                if (is_stable) {
                    target_found = true;
                    target_count++;
                }

                float area = static_cast<float>(rect.width * rect.height);
                if (area > best_target_area) {
                    best_target_area = area;
                    best_target_rect = rect;
                    best_target_width = rect.width;
                    target_confidence = avg_conf;  // Сохраняем confidence

                    Object temp_obj;
                    temp_obj.rect = rect;
                    temp_obj.label = obj.label;
                    temp_obj.prob = avg_conf;
                    SearchZone zone = calculateZone(temp_obj, frame.cols);
                    target_zone = zone.zone;
                    target_center_x = zone.center_x;
                }
            }

            vision_msgs::msg::Detection2D detection;
            detection.results.resize(1);
            detection.results[0].hypothesis.class_id = class_name;
            detection.results[0].hypothesis.score = avg_conf;

            detection.bbox.size_x = rect.width;
            detection.bbox.size_y = rect.height;
            detection.bbox.center.position.x = smoothed_center.x;
            detection.bbox.center.position.y = smoothed_center.y;

            detection_array.detections.push_back(detection);

            cv::Scalar color = getCategoryColorScalar(category);
            if (class_name == target_object_) {
                color = cv::Scalar(0, 0, 255);
            }

            cv::rectangle(frame, rect, color, 1);

            char label[128];
            if (params_.enable_tracking) {
                snprintf(label, sizeof(label), "#%d %s %.0f%% [%s]",
                         obj.id, class_name.c_str(), avg_conf * 100, category.c_str());
            } else if (show_category_) {
                snprintf(label, sizeof(label), "%s %.0f%% [%s]",
                         class_name.c_str(), avg_conf * 100, category.c_str());
            } else {
                snprintf(label, sizeof(label), "%s %.0f%%",
                         class_name.c_str(), avg_conf * 100);
            }

            int baseline = 0;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
            cv::rectangle(frame,
                         cv::Point(rect.x, rect.y - label_size.height - 10),
                         cv::Point(rect.x + label_size.width, rect.y),
                         color, -1);
            cv::putText(frame, label, cv::Point(rect.x, rect.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);

            if (params_.enable_tracking && obj.history.size() > 2) {
                for (size_t h = 1; h < obj.history.size(); h++) {
                    cv::Point p1(obj.history[h-1].x + obj.history[h-1].width/2,
                                 obj.history[h-1].y + obj.history[h-1].height/2);
                    cv::Point p2(obj.history[h].x + obj.history[h].width/2,
                                 obj.history[h].y + obj.history[h].height/2);
                    cv::line(frame, p1, p2, color, 2);
                }
            }

            if (!detected_classes.empty()) detected_classes += ", ";
            detected_classes += class_name;
        }

        // Publish detection array
        detections_pub_->publish(detection_array);

        // =====================================================================
        // AUTO-FOLLOW CONTROL — ПОЛНОСТЬЮ ПЕРЕПИСАН КАК В ОРИГИНАЛЕ ZRobot
        // =====================================================================
        static geometry_msgs::msg::Twist last_cmd;
        if (cmd_vel_pub_->get_subscription_count() > 0 && enable_auto_follow_) {
            auto now = std::chrono::steady_clock::now();

            // --- ГИСТЕРЕЗИС ПЕРЕКЛЮЧЕНИЯ МЕЖДУ РЕЖИМАМИ ---
            if (target_found) {
                last_detection_time_ = now;

                if (in_search_mode_) {
                    // Проверяем, прошло ли достаточно времени для выхода из поиска
                    auto search_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - search_start_time_).count();

                    if (search_elapsed > motor_params_.search_hysteresis_ms.load()) {
                        // Выходим из режима поиска
                        in_search_mode_ = false;
                        is_searching_ = false;
                        RCLCPP_INFO(this->get_logger(), "[Track] Target reacquired, resuming tracking");

                        // Очищаем историю скоростей для плавного перехода
                        speed_history_l_.clear();
                        speed_history_r_.clear();
                    }
                }

                was_target_found_ = true;
                has_been_detected_ = true;

                if (!in_search_mode_) {
                    last_known_zone_ = target_zone;
                    if (target_zone == "LEFT") {
                        search_direction_ = -1;
                    } else if (target_zone == "RIGHT") {
                        search_direction_ = 1;
                    }
                }
            } else {
                // Объект не найден
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - last_detection_time_).count();

                if (!in_search_mode_) {
                    if (was_target_found_ && elapsed > motor_params_.search_timeout_ms.load()) {
                        // Входим в режим поиска
                        in_search_mode_ = true;
                        search_start_time_ = now;
                        is_searching_ = true;

                        if (last_known_zone_ == "NONE" || last_known_zone_.empty()) {
                            search_direction_ = 1;
                        }

                        RCLCPP_INFO(this->get_logger(), "[Search] Target lost, searching %s",
                                    search_direction_ > 0 ? "RIGHT" : "LEFT");

                        // Очищаем историю для резкого перехода на поиск
                        speed_history_l_.clear();
                        speed_history_r_.clear();
                    } else if (!was_target_found_ && !has_been_detected_ && elapsed > 500) {
                        // При старте
                        in_search_mode_ = true;
                        search_start_time_ = now;
                        is_searching_ = true;
                        search_direction_ = 1;

                        RCLCPP_INFO(this->get_logger(), "[Search] Startup search RIGHT");
                    }
                }
            }

            // Расчет целевых скоростей (PWM 0-255)
            int target_left = 0;
            int target_right = 0;
            calculateMotorSpeedsProportional(target_center_x, frame.cols, target_found, best_target_width,
                                             target_left, target_right);

            // СГЛАЖИВАНИЕ СКОРОСТЕЙ
            int left_speed, right_speed;
            smoothSpeeds(target_left, target_right, left_speed, right_speed);

            // Конвертация PWM → нормализованный Twist
            double linear_norm, angular_norm;
            convertPWMtoTwist(left_speed, right_speed, linear_norm, angular_norm);

            geometry_msgs::msg::Twist cmd;
            cmd.linear.x = linear_norm * max_linear_speed_;
            cmd.angular.z = angular_norm * turn_speed_;

            // Clamp
            cmd.linear.x = std::max(-max_linear_speed_, std::min(max_linear_speed_, cmd.linear.x));
            cmd.angular.z = std::max(-turn_speed_, std::min(turn_speed_, cmd.angular.z));

            // Debug логирование раз в секунду
            static auto last_debug = std::chrono::steady_clock::now();
            if (now - last_debug > std::chrono::seconds(1)) {
                if (in_search_mode_) {
                    RCLCPP_INFO(this->get_logger(),
                                "[Search] Dir:%s L:%d R:%d | norm: L=%.2f A=%.2f%s",
                                search_direction_ > 0 ? "R" : "L",
                                left_speed, right_speed,
                                cmd.linear.x, cmd.angular.z,
                                target_found ? " (target visible)" : "");
                } else {
                    RCLCPP_INFO(this->get_logger(),
                                "[Track] Found:%s Zone:%s L:%d R:%d | norm: L=%.2f A=%.2f",
                                target_found ? "Y" : "N",
                                target_zone.c_str(),
                                left_speed, right_speed,
                                cmd.linear.x, cmd.angular.z);
                }
                last_debug = now;
            }

            last_cmd = cmd;
            cmd_vel_pub_->publish(cmd);
        }

        // Publish status message
        if (status_pub_->get_subscription_count() > 0) {
            std_msgs::msg::String status_msg;
            char status_buf[600];
            snprintf(status_buf, sizeof(status_buf),
                    "{\"target\":\"%s\",\"found\":%s,\"stable\":%s,\"zone\":\"%s\",\"count\":%d,"
                    "\"inference_time\":%.1f,\"fps\":%.1f,\"tracked\":%zu,\"classes\":\"%s\","
                    "\"searching\":%s,\"search_dir\":\"%s\",\"confidence\":%.2f,\"threshold\":%.2f}",
                    target_object_.c_str(),
                    target_found ? "true" : "false",
                    target_found ? "true" : "false",
                    target_zone.c_str(),
                    target_count,
                    inference_time_ms,
                    current_fps_,
                    tracked_objects.size(),
                    detected_classes.c_str(),
                    in_search_mode_ ? "true" : "false",
                    search_direction_ > 0 ? "RIGHT" : "LEFT",
                    target_confidence,
                    params_.conf_threshold);
            status_msg.data = status_buf;
            status_pub_->publish(status_msg);
        }

        // Publish tracked objects info
        if (tracked_pub_->get_subscription_count() > 0 && params_.enable_tracking) {
            std_msgs::msg::String tracked_msg;
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < tracked_objects.size(); i++) {
                if (i > 0) oss << ",";
                const auto& t = tracked_objects[i];
                cv::Point2f vel = t.getVelocity();
                oss << "{\"id\":" << t.id
                    << ",\"class\":\"" << t.class_name << "\""
                    << ",\"conf\":" << std::fixed << std::setprecision(2) << t.getAverageConfidence()
                    << ",\"vx\":" << vel.x << ",\"vy\":" << vel.y << "}";
            }
            oss << "]";
            tracked_msg.data = oss.str();
            tracked_pub_->publish(tracked_msg);
        }

        RCLCPP_DEBUG(this->get_logger(),
                    "Detected %zu objects, tracked %zu, target=%s, zone=%s, searching=%s",
                    objects.size(),
                    tracked_objects.size(),
                    target_found ? "YES" : "NO",
                    target_zone.c_str(),
                    in_search_mode_ ? "YES" : "NO");

        last_target_stable_ = target_found;
    }

    cv::Scalar getCategoryColorScalar(const std::string& category) {
        if (category == "person") return cv::Scalar(0, 0, 255);
        if (category == "vehicle") return cv::Scalar(0, 255, 0);
        if (category == "animal") return cv::Scalar(255, 0, 0);
        if (category == "food" || category == "fruit" || category == "vegetable")
            return cv::Scalar(0, 255, 255);
        if (category == "electronic" || category == "appliance")
            return cv::Scalar(255, 0, 255);
        return cv::Scalar(255, 255, 0);
    }

    // ROS2 members
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detections_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr tracked_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr target_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr confidence_sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    // YOLO and camera
    YOLOv8RKNN yolo_;
    cv::VideoCapture cap_;

    // State
    std::string target_object_;
    DetectionParams params_;
    MultiObjectTracker tracker_;
    PerformanceMonitor perf_monitor_;
    AdaptiveThresholdController threshold_controller_;
    DetectionStabilityController stability_controller_;
    ClassStabilityVerifier class_verifier_;
    std::unordered_map<int, PositionSmoother> position_smoothers_;

    bool adaptive_threshold_;
    bool show_category_;
    bool enable_auto_follow_;
    double max_linear_speed_;
    double turn_speed_;
    float base_conf_threshold_;
    bool last_target_stable_ = false;

    // Motor control parameters (атомарные для поткобезопасности)
    MotorControlParams motor_params_;

    // Состояние поиска/слежения
    std::chrono::steady_clock::time_point last_detection_time_;
    std::chrono::steady_clock::time_point search_start_time_;
    bool was_target_found_;
    bool in_search_mode_;
    std::atomic<bool> is_searching_;
    std::atomic<int> search_direction_;
    bool has_been_detected_;
    std::string last_known_zone_;

    // История скоростей для сглаживания
    std::deque<int> speed_history_l_;
    std::deque<int> speed_history_r_;

    std::atomic<int> frame_count_;
    rclcpp::Time last_fps_time_;
    double current_fps_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<YoloDetectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
