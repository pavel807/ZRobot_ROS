#ifndef OBJECT_TRACKER_H
#define OBJECT_TRACKER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>
#include <deque>
#include <cmath>

// Kalman Filter for object tracking
class KalmanFilter {
public:
    KalmanFilter() : initialized_(false) {}
    
    void init(const cv::Point2f& pos, const cv::Size2f& vel) {
        // State: [x, y, vx, vy, w, h]
        kf_ = cv::KalmanFilter(6, 4, 0, CV_32F);
        
        // Transition matrix - constant velocity model
        kf_.transitionMatrix = (cv::Mat_<float>(6, 6) <<
            1, 0, 1, 0, 0, 0,
            0, 1, 0, 1, 0, 0,
            0, 0, 1, 0, 0, 0,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1);
        
        // Measurement matrix
        kf_.measurementMatrix = (cv::Mat_<float>(4, 6) <<
            1, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1);
        
        // Process noise covariance
        kf_.processNoiseCov = cv::Mat::eye(6, 6, CV_32F) * 0.1f;
        
        // Measurement noise covariance
        kf_.measurementNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 0.5f;
        
        // Error covariance
        kf_.errorCovPost = cv::Mat::eye(6, 6, CV_32F) * 1.0f;
        
        // Initialize state
        kf_.statePost.at<float>(0) = pos.x;
        kf_.statePost.at<float>(1) = pos.y;
        kf_.statePost.at<float>(2) = vel.width;
        kf_.statePost.at<float>(3) = vel.height;
        kf_.statePost.at<float>(4) = 0;
        kf_.statePost.at<float>(5) = 0;
        
        initialized_ = true;
    }
    
    cv::Point2f predict() const {
        if (!initialized_) return cv::Point2f(0, 0);
        // Note: KalmanFilter::predict() modifies state, so we cast away const
        // This is safe because we're only reading the predicted state
        auto& kf_ref = const_cast<cv::KalmanFilter&>(kf_);
        kf_ref.predict();
        return cv::Point2f(
            kf_.statePost.at<float>(0),
            kf_.statePost.at<float>(1)
        );
    }
    
    cv::Point2f update(const cv::Point2f& measurement, float w, float h) {
        if (!initialized_) {
            init(measurement, cv::Size2f(w, h));
            return measurement;
        }
        
        cv::Mat measurement_mat = (cv::Mat_<float>(4, 1) <<
            measurement.x, measurement.y, w, h);
        
        kf_.correct(measurement_mat);
        return cv::Point2f(
            kf_.statePost.at<float>(0),
            kf_.statePost.at<float>(1)
        );
    }
    
    bool isInitialized() const { return initialized_; }
    
private:
    cv::KalmanFilter kf_;
    bool initialized_;
};

// Tracked object with history
struct TrackedObject {
    int id;
    int label;
    std::string class_name;
    KalmanFilter tracker;
    std::deque<cv::Rect> history;
    std::deque<float> confidence_history;
    int age;
    int consecutive_misses;
    cv::Point2f last_position;
    cv::Size2f last_size;
    bool is_confirmed;
    
    TrackedObject(int id_, int label_, const std::string& name,
                  const cv::Rect& rect, float conf)
        : id(id_), label(label_), class_name(name),
          age(0), consecutive_misses(0), is_confirmed(false) {
        history.push_back(rect);
        confidence_history.push_back(conf);
        last_position = cv::Point2f(rect.x + rect.width/2, rect.y + rect.height/2);
        last_size = cv::Size2f(rect.width, rect.height);
        tracker.init(last_position, cv::Size2f(0, 0));
    }
    
    void update(const cv::Rect& rect, float conf) {
        cv::Point2f center(rect.x + rect.width/2, rect.y + rect.height/2);
        tracker.update(center, rect.width, rect.height);
        
        history.push_back(rect);
        if (history.size() > 30) history.pop_front();
        
        confidence_history.push_back(conf);
        if (confidence_history.size() > 30) confidence_history.pop_front();
        
        last_position = center;
        last_size = cv::Size2f(rect.width, rect.height);
        age++;
        consecutive_misses = 0;
        
        if (age >= 3) is_confirmed = true;
    }
    
    void predict(cv::Rect& predicted_rect) const {
        if (history.empty()) {
            predicted_rect = cv::Rect(0, 0, 0, 0);
            return;
        }
        cv::Point2f pred = tracker.predict();
        predicted_rect = cv::Rect(
            pred.x - last_size.width/2,
            pred.y - last_size.height/2,
            last_size.width,
            last_size.height
        );
    }
    
    void updateSmoothed(const cv::Rect& rect, float conf) {
        cv::Point2f center(rect.x + rect.width/2, rect.y + rect.height/2);
        
        // Apply Kalman filter update
        tracker.update(center, rect.width, rect.height);
        
        // Smooth position with exponential moving average
        if (!history.empty()) {
            const cv::Rect& last = history.back();
            cv::Point2f last_center(last.x + last.width/2, last.y + last.height/2);
            float smoothing = 0.7f;
            center.x = smoothing * center.x + (1.0f - smoothing) * last_center.x;
            center.y = smoothing * center.y + (1.0f - smoothing) * last_center.y;
        }
        
        cv::Rect smoothed_rect(
            static_cast<int>(center.x - rect.width/2),
            static_cast<int>(center.y - rect.height/2),
            rect.width,
            rect.height
        );
        
        history.push_back(smoothed_rect);
        if (history.size() > 35) history.pop_front();
        
        confidence_history.push_back(conf);
        if (confidence_history.size() > 35) confidence_history.pop_front();
        
        last_position = center;
        last_size = cv::Size2f(rect.width, rect.height);
        age++;
        consecutive_misses = 0;
        
        // Require more frames for confirmation (improved stability)
        if (age >= 5) is_confirmed = true;
    }
    
    float getAverageConfidence() const {
        if (confidence_history.empty()) return 0.0f;
        float sum = 0.0f;
        for (float c : confidence_history) sum += c;
        return sum / confidence_history.size();
    }
    
    cv::Point2f getVelocity() const {
        if (history.size() < 2) return cv::Point2f(0, 0);
        const auto& curr = history.back();
        const auto& prev = history[history.size() - 2];
        return cv::Point2f(
            (curr.x + curr.width/2) - (prev.x + prev.width/2),
            (curr.y + curr.height/2) - (prev.y + prev.height/2)
        );
    }
};

// Multi-object tracker with Hungarian algorithm - Enhanced stability
class MultiObjectTracker {
public:
    MultiObjectTracker(float iou_threshold = 0.35f,      // Higher for stricter matching
                       int max_consecutive_misses = 10,    // Faster response to lost targets
                       float confidence_decay = 0.95f)     // Slower decay for stability
        : iou_threshold_(iou_threshold),
          max_consecutive_misses_(max_consecutive_misses),
          confidence_decay_(confidence_decay),
          next_id_(0) {}
    
    std::vector<TrackedObject> update(const std::vector<cv::Rect>& detections,
                                       const std::vector<int>& labels,
                                       const std::vector<std::string>& class_names,
                                       const std::vector<float>& confidences) {
        // Age existing tracks with decay
        for (auto& track : tracks_) {
            track.consecutive_misses++;
            if (!track.confidence_history.empty()) {
                track.confidence_history.push_back(
                    track.confidence_history.back() * confidence_decay_
                );
                if (track.confidence_history.size() > 35) {
                    track.confidence_history.pop_front();
                }
            }
        }
        
        // Build cost matrix (IoU based) with class matching bonus
        int n_tracks = tracks_.size();
        int n_detections = detections.size();
        
        std::vector<std::vector<float>> cost_matrix(n_tracks, 
                                                     std::vector<float>(n_detections, 0));
        
        for (int i = 0; i < n_tracks; i++) {
            for (int j = 0; j < n_detections; j++) {
                if (tracks_[i].label != labels[j]) {
                    cost_matrix[i][j] = 0;  // Different classes - no match
                } else {
                    float iou = computeIoU(tracks_[i], detections[j]);
                    // Add confidence bonus for better detections
                    cost_matrix[i][j] = iou * (0.7f + 0.3f * confidences[j]);
                }
            }
        }
        
        // Hungarian algorithm (greedy with priority queue)
        std::vector<bool> track_matched(n_tracks, false);
        std::vector<bool> detection_matched(n_detections, false);
        std::vector<std::pair<int, int>> matches;
        
        // Sort by highest cost first
        std::vector<std::tuple<float, int, int>> sorted_costs;
        for (int i = 0; i < n_tracks; i++) {
            for (int j = 0; j < n_detections; j++) {
                if (cost_matrix[i][j] > iou_threshold_) {
                    sorted_costs.emplace_back(cost_matrix[i][j], i, j);
                }
            }
        }
        std::sort(sorted_costs.begin(), sorted_costs.end(), 
                  std::greater<std::tuple<float, int, int>>());
        
        for (const auto& [cost, ti, di] : sorted_costs) {
            if (!track_matched[ti] && !detection_matched[di]) {
                matches.emplace_back(ti, di);
                track_matched[ti] = true;
                detection_matched[di] = true;
            }
        }
        
        // Update matched tracks with prediction smoothing
        for (const auto& [ti, di] : matches) {
            tracks_[ti].updateSmoothed(detections[di], confidences[di]);
        }
        
        // Create new tracks for unmatched detections
        for (int j = 0; j < n_detections; j++) {
            if (!detection_matched[j]) {
                tracks_.emplace_back(
                    next_id_++, labels[j], class_names[j],
                    detections[j], confidences[j]
                );
            }
        }
        
        // Remove lost tracks with lower confidence requirement
        tracks_.erase(
            std::remove_if(tracks_.begin(), tracks_.end(),
                [this](const TrackedObject& t) {
                    return t.consecutive_misses > max_consecutive_misses_ ||
                           t.getAverageConfidence() < 0.35f;
                }),
            tracks_.end()
        );
        
        return tracks_;
    }
    
    void clear() { tracks_.clear(); next_id_ = 0; }
    
    size_t getActiveTrackCount() const { return tracks_.size(); }
    
private:
    float computeIoU(const TrackedObject& track, const cv::Rect& detection) {
        cv::Rect pred_rect;
        track.predict(pred_rect);
        
        float x1 = std::max(pred_rect.x, detection.x);
        float y1 = std::max(pred_rect.y, detection.y);
        float x2 = std::min(pred_rect.br().x, detection.br().x);
        float y2 = std::min(pred_rect.br().y, detection.br().y);
        
        float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
        float union_area = pred_rect.area() + detection.area() - intersection;
        
        return union_area > 0 ? intersection / union_area : 0;
    }
    
    std::vector<TrackedObject> tracks_;
    float iou_threshold_;
    int max_consecutive_misses_;
    float confidence_decay_;
    int next_id_;
};

#endif // OBJECT_TRACKER_H
