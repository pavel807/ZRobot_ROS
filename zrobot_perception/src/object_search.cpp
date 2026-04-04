#include "object_search.h"
#include "colors.h"
#include <iostream>
#include <algorithm>
#include <cctype>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>

// ============================================================================
// COCO CLASS NAMES — 80 классов (как в оригинале)
// ============================================================================
const char* COCO_NAMES[80] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

// ============================================================================
// ЗОННАЯ ЛОГИКА
// ============================================================================
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

// ============================================================================
// НЕЧЁТКИЙ ПОИСК КЛАССОВ (как в оригинале ZRobot)
// ============================================================================
std::string toLower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return result;
}

bool caseInsensitiveCompare(const std::string& a, const std::string& b) {
    return toLower(a) == toLower(b);
}

// Расстояние Левенштейна для исправления опечаток
int levenshteinDistance(const std::string& s1, const std::string& s2) {
    int m = s1.length();
    int n = s2.length();
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));

    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (s1[i-1] == s2[j-1])
                dp[i][j] = dp[i-1][j-1];
            else
                dp[i][j] = 1 + std::min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]});
        }
    }
    return dp[m][n];
}

std::vector<SearchResult> findMatchingClasses(const std::string& query) {
    std::vector<SearchResult> results;
    std::string query_lower = toLower(query);
    query_lower.erase(0, query_lower.find_first_not_of(" \t\n\r"));
    query_lower.erase(query_lower.find_last_not_of(" \t\n\r") + 1);

    // 1. Проверка на число (ID класса)
    try {
        int num = std::stoi(query_lower);
        if (num >= 0 && num < 80) {
            results.push_back({COCO_NAMES[num], num, 0, "ID класса"});
            return results;
        }
    } catch (...) {}

    // 2. Точное совпадение (case-insensitive)
    for (int i = 0; i < 80; ++i) {
        if (caseInsensitiveCompare(query, COCO_NAMES[i])) {
            results.push_back({COCO_NAMES[i], i, 0, "точное совпадение"});
            return results;
        }
    }

    // 3. Поиск по подстроке (начинается с...)
    for (int i = 0; i < 80; ++i) {
        std::string class_lower = toLower(COCO_NAMES[i]);
        if (class_lower.find(query_lower) == 0) {
            results.push_back({COCO_NAMES[i], i, 1, "начинается с '" + query + "'"});
        }
    }

    // 4. Поиск по подстроке (содержит...)
    for (int i = 0; i < 80; ++i) {
        std::string class_lower = toLower(COCO_NAMES[i]);
        if (class_lower.find(query_lower) != std::string::npos) {
            // Проверяем, не добавили ли уже (из предыдущего шага)
            bool exists = false;
            for (const auto& r : results) {
                if (r.index == i) { exists = true; break; }
            }
            if (!exists) {
                results.push_back({COCO_NAMES[i], i, 2, "содержит '" + query + "'"});
            }
        }
    }

    // 5. Нечеткий поиск (исправление опечаток)
    if (results.empty()) {
        std::vector<std::pair<int, int>> distances; // <distance, index>
        for (int i = 0; i < 80; ++i) {
            int dist = levenshteinDistance(query_lower, toLower(COCO_NAMES[i]));
            if (dist <= 3 && query_lower.length() > 2) { // не ищем опечатки в коротких словах
                distances.push_back({dist, i});
            }
        }
        std::sort(distances.begin(), distances.end());

        int max_suggestions = 5;
        for (size_t i = 0; i < distances.size() && i < static_cast<size_t>(max_suggestions); ++i) {
            int dist = distances[i].first;
            int idx = distances[i].second;
            std::string type = "похоже (ошибка в " + std::to_string(dist) + " симв.)";
            results.push_back({COCO_NAMES[idx], idx, 3 + dist, type});
        }
    }

    return results;
}

void printColoredClassList() {
    std::cout << BOLD << "Доступные классы (" << BLUE << "0-79" << RESET << "):" << RESET << "\n";
    for (int i = 0; i < 80; ++i) {
        if (i % 4 == 0) std::cout << "  ";
        std::cout << CYAN << std::setw(2) << i << RESET << ":"
                  << std::setw(15) << std::left << COCO_NAMES[i] << RESET;
        if ((i + 1) % 4 == 0) std::cout << "\n";
        else std::cout << "  ";
    }
    std::cout << "\n";
}
