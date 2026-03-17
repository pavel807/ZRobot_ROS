#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <geometry_msgs/msg/point.hpp>
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
// ADVANCED DETECTION PARAMETERS
// ============================================================================
struct DetectionParams {
    float conf_threshold = 0.25f;
    float nms_threshold = 0.45f;
    float min_area_ratio = 0.001f;
    float max_area_ratio = 0.5f;
    bool enable_tracking = true;
    bool temporal_filtering = true;
    int history_size = 30;
    float velocity_weight = 0.3f;
    float confidence_decay = 0.9f;
    int max_consecutive_misses = 15;
    float iou_threshold = 0.3f;
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
            // Increase threshold to reduce detections
            conf_threshold_ = std::min(0.8f, conf_threshold_ + 0.02f);
        } else if (current_fps > target_fps * 1.2f) {
            // Decrease threshold for more detections
            conf_threshold_ = std::max(0.1f, conf_threshold_ - 0.01f);
        }
    }
    
    float getThreshold() const { return conf_threshold_; }
    void reset(float value) { conf_threshold_ = value; }
    
private:
    float conf_threshold_ = 0.25f;
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
        this->declare_parameter<float>("conf_threshold", 0.25f);
        this->declare_parameter<std::string>("target_object", "person");
        this->declare_parameter<bool>("enable_tracking", true);
        this->declare_parameter<bool>("adaptive_threshold", true);
        this->declare_parameter<bool>("show_category", true);
        
        // Get parameters
        std::string model_path;
        int camera_id, width, height, fps;
        float conf_threshold;
        bool enable_tracking, adaptive_threshold, show_category;
        
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
        
        params_.conf_threshold = conf_threshold;
        params_.enable_tracking = enable_tracking;
        adaptive_threshold_ = adaptive_threshold;
        show_category_ = show_category;
        
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
        target_object_ = msg->data;
        RCLCPP_INFO(this->get_logger(), "Target changed to: %s", target_object_.c_str());
        // Reset tracker on target change
        if (params_.enable_tracking) {
            tracker_.clear();
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
        
        // Update FPS counter on image
        drawOverlay(frame);
        
        // Publish processed image
        if (image_pub_->get_subscription_count() > 0) {
            std_msgs::msg::Header header;
            header.stamp = this->now();
            header.frame_id = "camera";
            auto img_msg = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, frame).toImageMsg();
            image_pub_->publish(*img_msg);
        }
    }
    
    void drawOverlay(cv::Mat& frame) {
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
            snprintf(track_text, sizeof(track_text), "Tracked: %zu objects", 
                     tracker_.getActiveTrackCount());
            cv::putText(frame, track_text, cv::Point(10, 110),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 255), 2);
        }
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
        std::string detected_classes;

        // Process tracked objects or raw detections based on mode
        const auto& active_objects = tracked_objects;

        for (size_t i = 0; i < active_objects.size(); i++) {
            const auto& obj = active_objects[i];
            const std::string& class_name = obj.class_name;

            // Get category
            auto cat_it = ID_TO_CATEGORY.find(obj.label);
            std::string category = (cat_it != ID_TO_CATEGORY.end()) ? cat_it->second : "unknown";

            // Check if this is the target object
            if (class_name == target_object_) {
                target_found = true;
                target_count++;
                cv::Rect rect = obj.history.back();
                // Create temporary Object for calculateZone
                Object temp_obj;
                temp_obj.rect = rect;
                temp_obj.label = obj.label;
                temp_obj.prob = obj.getAverageConfidence();
                SearchZone zone = calculateZone(temp_obj, frame.cols);
                target_zone = zone.zone;
                target_center_x = zone.center_x;
            }

            // Create detection message
            vision_msgs::msg::Detection2D detection;
            detection.results.resize(1);
            detection.results[0].hypothesis.class_id = class_name;
            detection.results[0].hypothesis.score = obj.getAverageConfidence();

            cv::Rect rect = obj.history.back();
            detection.bbox.size_x = rect.width;
            detection.bbox.size_y = rect.height;
            detection.bbox.center.position.x = rect.x + rect.width / 2.0;
            detection.bbox.center.position.y = rect.y + rect.height / 2.0;

            detection_array.detections.push_back(detection);

            // Draw on frame with category color
            cv::Scalar color = getCategoryColorScalar(category);
            if (class_name == target_object_) {
                color = cv::Scalar(0, 0, 255);  // Red for target
            }

            // Draw bounding box with tracking ID
            cv::rectangle(frame, rect, color, 2);

            char label[128];
            float confidence = obj.getAverageConfidence();
            if (params_.enable_tracking) {
                snprintf(label, sizeof(label), "#%d %s %.0f%% [%s]",
                         obj.id, class_name.c_str(), confidence * 100, category.c_str());
            } else if (show_category_) {
                snprintf(label, sizeof(label), "%s %.0f%% [%s]",
                         class_name.c_str(), confidence * 100, category.c_str());
            } else {
                snprintf(label, sizeof(label), "%s %.0f%%",
                         class_name.c_str(), confidence * 100);
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
        
        // Publish status message
        if (status_pub_->get_subscription_count() > 0) {
            std_msgs::msg::String status_msg;
            char status_buf[512];
            snprintf(status_buf, sizeof(status_buf),
                    "{\"target\":\"%s\",\"found\":%s,\"zone\":\"%s\",\"count\":%d,"
                    "\"inference_time\":%.1f,\"fps\":%.1f,\"tracked\":%zu,\"classes\":\"%s\"}",
                    target_object_.c_str(),
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
    
    bool adaptive_threshold_;
    bool show_category_;
    float base_conf_threshold_;
    
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
