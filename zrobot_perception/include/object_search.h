#ifndef OBJECT_SEARCH_H
#define OBJECT_SEARCH_H

#include <string>
#include <vector>
#include "object_struct.h"

// COCO class names (80 classes)
extern const char* COCO_NAMES[80];

// Search zone calculation
struct SearchZone {
    std::string zone;  // "LEFT", "CENTER", "RIGHT", "NONE"
    float center_x;
    float deviation;
};

// Object search functionality (как в оригинале ZRobot)
struct SearchResult {
    std::string name;
    int index;
    int priority;    // 0 = точное, 1 = начинается с, 2 = содержит, 3+ = нечёткий
    std::string match_type;
};

SearchZone calculateZone(const Object& obj, int frame_width);
std::string getZoneName(float center_x, int frame_width);

// Нечёткий поиск классов (из оригинала ZRobot)
std::string toLower(const std::string& str);
bool caseInsensitiveCompare(const std::string& a, const std::string& b);
int levenshteinDistance(const std::string& s1, const std::string& s2);
std::vector<SearchResult> findMatchingClasses(const std::string& query);
void printColoredClassList();

#endif // OBJECT_SEARCH_H
