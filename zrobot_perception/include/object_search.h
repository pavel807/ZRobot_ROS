#ifndef OBJECT_SEARCH_H
#define OBJECT_SEARCH_H

#include <string>
#include <vector>
#include "object_struct.h"

// Search zone calculation
struct SearchZone {
    std::string zone;  // "LEFT", "CENTER", "RIGHT", "NONE"
    float center_x;
    float deviation;
};

SearchZone calculateZone(const Object& obj, int frame_width);
std::string getZoneName(float center_x, int frame_width);

#endif // OBJECT_SEARCH_H
