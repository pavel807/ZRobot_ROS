#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/string.hpp>
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

// Keep COCO_NAMES for backward compatibility
const std::vector<std::string> COCO_NAMES = []() {
    std::vector<std::string> names;
    for (const auto& cls : COCO_CLASSES) {
        names.push_back(cls.name);
    }
    return names;
}();

// ============================================================================
// ADVANCED DETECTION PARAMETERS - OPTIMIZED FOR STABILITY
// ============================================================================
struct DetectionParams {
    float conf_threshold = 0.47f;          // 47% threshold for better accuracy
    float nms_threshold = 0.4f;              // Stricter NMS for fewer duplicates
    float min_area_ratio = 0.002f;           // Filter tiny detections
    float max_area_ratio = 0.45f;            // Filter oversized detections
    bool enable_tracking = true;
    bool temporal_filtering = true;
    int history_size = 35;                   // Larger history for smoother tracking
    float velocity_weight = 0.25f;           // Lower velocity weight for stability
    float confidence_decay = 0.95f;          // Slower decay = more stable confidence
    int max_consecutive_misses = 10;         // Faster response to lost targets
    float iou_threshold = 0.35f;             // Higher IoU threshold for stricter matching
    int min_confirm_frames = 5;              // Need 5 frames to confirm detection
    float hysteresis_margin = 0.05f;         // 5% margin for detection stability
    int class_stability_window = 5;          // Check class over 5 frames
    float position_smoothing = 0.6f;         // Smoothing factor for position (0-1)
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
            conf_threshold_ = std::max(0.35f, conf_threshold_ - 0.01f);
        }
    }
    
    float getThreshold() const { return conf_threshold_; }
    void reset(float value) { conf_threshold_ = value; }
    
private:
    float conf_threshold_ = 0.47f;
};

// ============================================================================
// DETECTION STABILITY CONTROLLER - Prevents flickering
// ============================================================================
class DetectionStabilityController {
public:
    DetectionStabilityController() : last_detection_state_(false), stable_count_(0), unstable_count_(0) {}
    
    bool isStableDetection(bool current_detected, float confidence) {
        if (current_detected && confidence >= 0.50f) {
            stable_count_++;
            unstable_count_ = 0;
            if (stable_count_ >= 3) {
                last_detection_state_ = true;
                return true;
            }
        } else if (!current_detected) {
            unstable_count_++;
            stable_count_ = 0;
            if (unstable_count_ >= 4) {
                last_detection_state_ = false;
            }
        } else if (confidence >= 0.47f && confidence < 0.50f) {
            stable_count_++;
            unstable_count_ = 0;
            if (stable_count_ >= 5) {
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
// CLASS STABILITY VERIFIER - Prevents class confusion
// ============================================================================
class ClassStabilityVerifier {
public:
    ClassStabilityVerifier(int window_size = 5) : window_size_(window_size) {}
    
    std::string getStableClass(const std::string& detected_class, float confidence) {
        if (confidence < 0.55f) {
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
        
        if (max_count >= 3) {
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
    int window_size_;
};

// ============================================================================
// POSITION SMOOTHER - Reduces jitter
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
// YOLO DETECTOR NODE
// ============================================================================
class YoloDetectorNode : public rclcpp::Node {
public:
    YoloDetectorNode() : Node("yolo_detector") {
        RCLCPP_INFO(this->get_logger(), "Initializing Advanced YOLO Detector...");
        RCLCPP_INFO(this->get_logger(), "COCO Classes: %zu objects in %zu categories", 
                    COCO_CLASSES.size(), COCO_CATEGORIES.size());
        
        // Declare parameters
        this->declare_parameter<std::string>("model_path", "models/yolov8s.rknn");
        this->declare_parameter<int>("camera_id", 1);
        this->declare_parameter<int>("width", 640);
        this->declare_parameter<int>("height", 640);
        this->declare_parameter<int>("fps", 30);
        this->declare_parameter<float>("conf_threshold", 0.47f);
        this->declare_parameter<std::string>("target_object", "person");
        this->declare_parameter<bool>("enable_tracking", true);
        this->declare_parameter<bool>("adaptive_threshold", false);
        this->declare_parameter<bool>("show_category", true);
        this->declare_parameter<bool>("enable_auto_follow", true);
        this->declare_parameter<double>("max_linear_speed", 0.3);
        this->declare_parameter<double>("turn_speed", 0.5);
        
        // Get parameters
        std::string model_path;
        int camera_id, width, height, fps;
        float conf_threshold;
        bool enable_tracking, adaptive_threshold, show_category;
        bool enable_auto_follow;
        double max_linear_speed, turn_speed;
        
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
        
        params_.conf_threshold = conf_threshold;
        params_.enable_tracking = enable_tracking;
        adaptive_threshold_ = adaptive_threshold;
        show_category_ = show_category;
        enable_auto_follow_ = enable_auto_follow;
        max_linear_speed_ = max_linear_speed;
        turn_speed_ = turn_speed;
        
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
        
        // Create timer for detection loop
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000 / fps),
            std::bind(&YoloDetectorNode::detectLoop, this)
        );
        
        frame_count_ = 0;
        last_fps_time_ = this->now();
        current_fps_ = 0.0;
        base_conf_threshold_ = conf_threshold;
        
        RCLCPP_INFO(this->get_logger(), "Advanced YOLO Detector node started");
        RCLCPP_INFO(this->get_logger(), "Target object: %s | Tracking: %s | Adaptive: %s",
                    target_object_.c_str(), 
                    enable_tracking ? "ON" : "OFF",
                    adaptive_threshold ? "ON" : "OFF");
    }

private:
    void targetCallback(const std_msgs::msg::String::SharedPtr msg) {
        if (target_object_ != msg->data) {
            target_object_ = msg->data;
            RCLCPP_INFO(this->get_logger(), "Target changed to: %s - resetting stability", target_object_.c_str());
            // Reset all stability components on target change
            if (params_.enable_tracking) {
                tracker_.clear();
            }
            stability_controller_.reset();
            class_verifier_.reset();
            position_smoothers_.clear();
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
            
            // Adaptive threshold control
            if (adaptive_threshold_) {
                threshold_controller_.update(current_fps_);
            }
            
            frame_count_ = 0;
            last_fps_time_ = now;
        }
        
        // Publish detections
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
        // Performance stats
        std::string stats = perf_monitor_.getStats();
        cv::putText(frame, stats, cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        // Target info
        char target_text[64];
        snprintf(target_text, sizeof(target_text), "Target: %s", target_object_.c_str());
        cv::putText(frame, target_text, cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
        
        // Threshold info
        char thresh_text[64];
        snprintf(thresh_text, sizeof(thresh_text), "Conf: %.2f%s", 
                 adaptive_threshold_ ? threshold_controller_.getThreshold() : params_.conf_threshold,
                 adaptive_threshold_ ? " (adaptive)" : "");
        cv::putText(frame, thresh_text, cv::Point(10, 85),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 200, 0), 2);
        
        // Tracker info
        if (params_.enable_tracking) {
            char track_text[64];
            snprintf(track_text, sizeof(track_text), "Tracked: %zu | Stable: %s", 
                     tracker_.getActiveTrackCount(),
                     target_is_stable ? "YES" : "NO");
            cv::putText(frame, track_text, cv::Point(10, 110),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 255), 2);
        }
        
        // Stability status indicator
        std::string stability_status = target_is_stable ? "STABLE" : "SEARCHING";
        cv::Scalar stability_color = target_is_stable ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
        cv::putText(frame, stability_status, cv::Point(frame.cols - 100, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, stability_color, 2);
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
        int target_count = 0;
        cv::Rect best_target_rect;
        float best_target_area = 0.0f;
        std::string detected_classes;

        // Process tracked objects or raw detections based on mode
        const auto& active_objects = tracked_objects;

        for (size_t i = 0; i < active_objects.size(); i++) {
            const auto& obj = active_objects[i];
            std::string class_name = obj.class_name;
            
            // Apply class stability verification to prevent misclassification
            float avg_conf = obj.getAverageConfidence();
            class_name = class_verifier_.getStableClass(class_name, avg_conf);
            
            // Apply position smoothing for smoother tracking
            cv::Rect rect = obj.history.back();
            cv::Point2f center(rect.x + rect.width/2.0f, rect.y + rect.height/2.0f);
            
            if (position_smoothers_.find(obj.id) == position_smoothers_.end()) {
                position_smoothers_[obj.id] = PositionSmoother(params_.position_smoothing);
            }
            cv::Point2f smoothed_center = position_smoothers_[obj.id].smooth(center);
            
            rect.x = static_cast<int>(smoothed_center.x - rect.width/2.0f);
            rect.y = static_cast<int>(smoothed_center.y - rect.height/2.0f);
            rect = rect & cv::Rect(0, 0, frame.cols, frame.rows);

            // Get category
            auto cat_it = ID_TO_CATEGORY.find(obj.label);
            std::string category = (cat_it != ID_TO_CATEGORY.end()) ? cat_it->second : "unknown";

            // Check if this is the target object (with stability check)
            if (class_name == target_object_) {
                // Apply detection stability check
                bool is_stable = stability_controller_.isStableDetection(true, avg_conf);
                if (is_stable) {
                    target_found = true;
                    target_count++;
                }
                
                float area = static_cast<float>(rect.width * rect.height);
                if (area > best_target_area) {
                    best_target_area = area;
                    best_target_rect = rect;
                    // Create temporary Object for calculateZone
                    Object temp_obj;
                    temp_obj.rect = rect;
                    temp_obj.label = obj.label;
                    temp_obj.prob = avg_conf;
                    SearchZone zone = calculateZone(temp_obj, frame.cols);
                    target_zone = zone.zone;
                    target_center_x = zone.center_x;
                }
            }

            // Create detection message
            vision_msgs::msg::Detection2D detection;
            detection.results.resize(1);
            detection.results[0].hypothesis.class_id = class_name;
            detection.results[0].hypothesis.score = avg_conf;

            detection.bbox.size_x = rect.width;
            detection.bbox.size_y = rect.height;
            detection.bbox.center.position.x = smoothed_center.x;
            detection.bbox.center.position.y = smoothed_center.y;

            detection_array.detections.push_back(detection);

            // Draw on frame with category color
            cv::Scalar color = getCategoryColorScalar(category);
            if (class_name == target_object_) {
                color = cv::Scalar(0, 0, 255);  // Red for target
            }

            // Draw bounding box with tracking ID
            cv::rectangle(frame, rect, color, 2);

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
            
            // Draw trajectory for tracked objects
            if (params_.enable_tracking && obj.history.size() > 2) {
                for (size_t h = 1; h < obj.history.size(); h++) {
                    cv::Point p1(obj.history[h-1].x + obj.history[h-1].width/2,
                                 obj.history[h-1].y + obj.history[h-1].height/2);
                    cv::Point p2(obj.history[h].x + obj.history[h].width/2,
                                 obj.history[h].y + obj.history[h].height/2);
                    cv::line(frame, p1, p2, color, 2);
                }
            }
            
            // Build detected classes string
            if (!detected_classes.empty()) detected_classes += ", ";
            detected_classes += class_name;
        }
        
        // Publish detection array
        detections_pub_->publish(detection_array);

        // Auto-follow control with stability - 3x3 grid (Left/Center/Right x Near/Mid/Far)
        static geometry_msgs::msg::Twist last_cmd;
        if (cmd_vel_pub_->get_subscription_count() > 0 && enable_auto_follow_) {
            geometry_msgs::msg::Twist cmd;
            
            // Only follow if target is stable (detected consistently)
            if (target_found && best_target_area > 0.0f) {
                // Distance estimation by bbox height ratio
                float height_ratio = static_cast<float>(best_target_rect.height) /
                                     static_cast<float>(frame.rows);

                // Speed bands based on distance with hysteresis
                const float min_speed_pwm = 115.0f;
                const float max_speed_pwm = 200.0f;
                float target_pwm = 0.0f;
                
                // Add hysteresis to prevent speed oscillation
                static float last_height_ratio = 0.0f;
                if (height_ratio > 0.45f) {          // Near
                    target_pwm = min_speed_pwm + 20.0f;
                } else if (height_ratio > 0.30f) {  // Mid-Near transition
                    target_pwm = (last_height_ratio > 0.45f) ? 
                                 min_speed_pwm + 20.0f : min_speed_pwm + 35.0f;
                } else if (height_ratio > 0.22f) {  // Mid
                    target_pwm = min_speed_pwm + 50.0f;
                } else if (height_ratio > 0.15f) {  // Mid-Far transition
                    target_pwm = (last_height_ratio > 0.22f) ?
                                 min_speed_pwm + 50.0f : max_speed_pwm - 30.0f;
                } else {                             // Far
                    target_pwm = max_speed_pwm - 15.0f;
                }
                last_height_ratio = height_ratio;
                
                // Convert desired PWM band into normalized linear speed (0..1)
                float linear_norm = target_pwm / max_speed_pwm;
                linear_norm = std::max(0.0f, std::min(1.0f, linear_norm));

                cmd.linear.x = linear_norm * static_cast<float>(max_linear_speed_);

                // Proportional angular control based on target offset
                float frame_center = static_cast<float>(frame.cols) / 2.0f;
                float target_center = best_target_rect.x + best_target_rect.width / 2.0f;
                float offset_ratio = (target_center - frame_center) / frame_center;
                
                // Apply proportional turning with deadzone
                float turn_deadzone = 0.08f;
                if (std::abs(offset_ratio) > turn_deadzone) {
                    float adjusted_offset = offset_ratio - (offset_ratio > 0 ? turn_deadzone : -turn_deadzone);
                    cmd.angular.z = static_cast<double>(turn_speed_) * adjusted_offset * 1.5;
                    cmd.angular.z = std::max(-turn_speed_, std::min(turn_speed_, cmd.angular.z));
                } else {
                    cmd.angular.z = 0.0;
                }
                
                // Smooth command transitions
                cmd.linear.x = 0.8f * last_cmd.linear.x + 0.2f * cmd.linear.x;
                cmd.angular.z = 0.7f * last_cmd.angular.z + 0.3f * cmd.angular.z;
                
            } else {
                // No stable target: gradual stop with momentum
                cmd.linear.x = 0.0f;
                cmd.angular.z = 0.0f;
            }
            last_cmd = cmd;
            cmd_vel_pub_->publish(cmd);
        }

        // Publish status message with stability info
        if (status_pub_->get_subscription_count() > 0) {
            std_msgs::msg::String status_msg;
            char status_buf[512];
            snprintf(status_buf, sizeof(status_buf),
                    "{\"target\":\"%s\",\"found\":%s,\"stable\":%s,\"zone\":\"%s\",\"count\":%d,"
                    "\"inference_time\":%.1f,\"fps\":%.1f,\"tracked\":%zu,\"classes\":\"%s\"}",
                    target_object_.c_str(),
                    target_found ? "true" : "false",
                    target_found ? "true" : "false",
                    target_zone.c_str(),
                    target_count,
                    inference_time_ms,
                    current_fps_,
                    tracked_objects.size(),
                    detected_classes.c_str());
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
                    "Detected %zu objects, tracked %zu, target=%s, zone=%s",
                    objects.size(),
                    tracked_objects.size(),
                    target_found ? "YES" : "NO",
                    target_zone.c_str());
        
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
