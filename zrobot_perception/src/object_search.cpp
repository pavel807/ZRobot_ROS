#include "object_search.h"

std::string getZoneName(float center_x, int frame_width) {
    const int zone_width = frame_width / 3;
    const int double_zone_width = 2 * zone_width;
    
    if (center_x < zone_width) {
        return "LEFT";
    } else if (center_x < double_zone_width) {
        return "CENTER";
    } else {
        return "RIGHT";
    }
}

SearchZone calculateZone(const Object& obj, int frame_width) {
    SearchZone zone;
    zone.center_x = obj.rect.x + obj.rect.width * 0.5f;
    zone.deviation = fabs(zone.center_x - frame_width / 2.0f) / (frame_width / 2.0f);
    zone.zone = getZoneName(zone.center_x, frame_width);
    return zone;
}
