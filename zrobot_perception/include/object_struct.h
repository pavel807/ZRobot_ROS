#ifndef OBJECT_STRUCT_H
#define OBJECT_STRUCT_H

#include <opencv2/opencv.hpp>
#include <utility>

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
    
    std::vector<std::pair<int, float>> multi_labels;
    
    float cx() const { return rect.x + rect.width * 0.5f; }
    float cy() const { return rect.y + rect.height * 0.5f; }
};

#endif // OBJECT_STRUCT_H
