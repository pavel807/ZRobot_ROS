#ifndef COCO_CLASSES_H
#define COCO_CLASSES_H

#include <vector>
#include <string>
#include <unordered_map>

// COCO Dataset - 80 Classes with Categories and Descriptions
// Full list: https://cocodataset.org/#home

struct CocoClass {
    int id;
    std::string name;
    std::string category;
    std::string description;
};

// Complete COCO 80 classes with categories
const std::vector<CocoClass> COCO_CLASSES = {
    // PERSON (0)
    {0, "person", "person", "Human person"},
    
    // VEHICLE (1-7)
    {1, "bicycle", "vehicle", "Two-wheeled vehicle"},
    {2, "car", "vehicle", "Four-wheeled automobile"},
    {3, "motorcycle", "vehicle", "Two-wheeled motor vehicle"},
    {4, "airplane", "vehicle", "Aircraft"},
    {5, "bus", "vehicle", "Large passenger vehicle"},
    {6, "train", "vehicle", "Railway vehicle"},
    {7, "truck", "vehicle", "Large goods vehicle"},
    {8, "boat", "vehicle", "Watercraft"},
    
    // OUTDOOR (8-13)
    {9, "traffic light", "outdoor", "Traffic signal"},
    {10, "fire hydrant", "outdoor", "Fire water supply"},
    {11, "stop sign", "outdoor", "Traffic stop sign"},
    {12, "parking meter", "outdoor", "Parking payment device"},
    {13, "bench", "outdoor", "Sitting furniture"},
    {14, "bird", "animal", "Flying bird"},
    
    // ANIMAL (14-22)
    {15, "cat", "animal", "Domestic cat"},
    {16, "dog", "animal", "Domestic dog"},
    {17, "horse", "animal", "Equine animal"},
    {18, "sheep", "animal", "Woolly livestock"},
    {19, "cow", "animal", "Bovine livestock"},
    {20, "elephant", "animal", "Large mammal"},
    {21, "bear", "animal", "Large carnivore"},
    {22, "zebra", "animal", "Striped equine"},
    {23, "giraffe", "animal", "Long-necked mammal"},
    
    // ACCESSORY (23-28)
    {24, "backpack", "accessory", "Carrying bag"},
    {25, "umbrella", "accessory", "Rain protection"},
    {26, "handbag", "accessory", "Hand carried bag"},
    {27, "tie", "accessory", "Neck clothing"},
    {28, "suitcase", "accessory", "Travel luggage"},
    
    // SPORTS (28-34)
    {29, "frisbee", "sports", "Flying disc"},
    {30, "skis", "sports", "Snow sliding equipment"},
    {31, "snowboard", "sports", "Snow sliding board"},
    {32, "sports ball", "sports", "Spherical sports object"},
    {33, "kite", "sports", "Flying toy"},
    {34, "baseball bat", "sports", "Baseball equipment"},
    {35, "baseball glove", "sports", "Baseball catching glove"},
    {36, "skateboard", "sports", "Wheeled board"},
    {37, "surfboard", "sports", "Wave riding board"},
    {38, "tennis racket", "sports", "Tennis equipment"},
    
    // CONTAINER (38-43)
    {39, "bottle", "container", "Liquid container"},
    {40, "wine glass", "container", "Drinking glass"},
    {41, "cup", "container", "Drinking cup"},
    {42, "fork", "utensil", "Eating utensil"},
    {43, "knife", "utensil", "Cutting utensil"},
    {44, "spoon", "utensil", "Eating spoon"},
    {45, "bowl", "container", "Food container"},
    
    // FRUIT (45-49)
    {46, "banana", "fruit", "Yellow curved fruit"},
    {47, "apple", "fruit", "Round red/green fruit"},
    {48, "sandwich", "food", "Bread with filling"},
    {49, "orange", "fruit", "Citrus fruit"},
    {50, "broccoli", "vegetable", "Green vegetable"},
    {51, "carrot", "vegetable", "Orange root vegetable"},
    {52, "hot dog", "food", "Sausage in bun"},
    {53, "pizza", "food", "Italian dish"},
    {54, "donut", "food", "Fried dough pastry"},
    {55, "cake", "food", "Sweet dessert"},
    
    // FURNITURE (55-61)
    {56, "chair", "furniture", "Sitting furniture"},
    {57, "couch", "furniture", "Sofa furniture"},
    {58, "potted plant", "furniture", "Indoor plant"},
    {59, "bed", "furniture", "Sleeping furniture"},
    {60, "dining table", "furniture", "Eating table"},
    {61, "toilet", "furniture", "Bathroom fixture"},
    
    // ELECTRONIC (61-68)
    {62, "tv", "electronic", "Television"},
    {63, "laptop", "electronic", "Portable computer"},
    {64, "mouse", "electronic", "Computer mouse"},
    {65, "remote", "electronic", "Remote control"},
    {66, "keyboard", "electronic", "Computer keyboard"},
    {67, "cell phone", "electronic", "Mobile phone"},
    
    // APPLIANCE (67-72)
    {68, "microwave", "appliance", "Cooking appliance"},
    {69, "oven", "appliance", "Cooking oven"},
    {70, "toaster", "appliance", "Bread toasting appliance"},
    {71, "sink", "appliance", "Washing sink"},
    {72, "refrigerator", "appliance", "Cooling appliance"},
    
    // INDOOR (72-79)
    {73, "book", "indoor", "Reading book"},
    {74, "clock", "indoor", "Time device"},
    {75, "vase", "indoor", "Decorative container"},
    {76, "scissors", "indoor", "Cutting tool"},
    {77, "teddy bear", "indoor", "Stuffed toy"},
    {78, "hair drier", "indoor", "Hair drying device"},
    {79, "toothbrush", "indoor", "Dental hygiene tool"}
};

// Category mappings for filtering
const std::vector<std::string> COCO_CATEGORIES = {
    "person", "vehicle", "animal", "outdoor", 
    "accessory", "sports", "container", "utensil",
    "fruit", "vegetable", "food", "furniture",
    "electronic", "appliance", "indoor"
};

// Quick lookup maps
inline const std::unordered_map<std::string, int> createClassToIdMap() {
    std::unordered_map<std::string, int> map;
    for (const auto& cls : COCO_CLASSES) {
        map[cls.name] = cls.id;
    }
    return map;
}

inline const std::unordered_map<int, std::string> createIdToCategoryMap() {
    std::unordered_map<int, std::string> map;
    for (const auto& cls : COCO_CLASSES) {
        map[cls.id] = cls.category;
    }
    return map;
}

// Pre-computed maps
static const std::unordered_map<std::string, int> CLASS_TO_ID = createClassToIdMap();
static const std::unordered_map<int, std::string> ID_TO_CATEGORY = createIdToCategoryMap();

// Get category color for visualization
inline const char* getCategoryColor(const std::string& category) {
    if (category == "person") return "255, 0, 0";      // Red
    if (category == "vehicle") return "0, 255, 0";     // Green
    if (category == "animal") return "0, 0, 255";      // Blue
    if (category == "food" || category == "fruit" || category == "vegetable") return "255, 255, 0"; // Yellow
    if (category == "electronic" || category == "appliance") return "255, 0, 255"; // Magenta
    return "0, 255, 255";  // Cyan for others
}

#endif // COCO_CLASSES_H
