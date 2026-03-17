#ifndef OBJECT_STRUCT_H
#define OBJECT_STRUCT_H

#include <opencv2/opencv.hpp>

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

#endif // OBJECT_STRUCT_H
