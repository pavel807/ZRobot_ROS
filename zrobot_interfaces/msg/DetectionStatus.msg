# Detection status message
# Published by yolo_detector_node

string target_object          # Target object class name
bool target_found             # True if target is detected
string zone                   # "LEFT", "CENTER", "RIGHT", "NONE"
int32 target_count            # Number of detected targets
float32 confidence            # Confidence threshold
float32 inference_time_ms     # Inference time in milliseconds
int32 fps                     # Current FPS
int32 frame_width             # Frame width
int32 frame_height            # Frame height
float32 target_center_x       # X coordinate of target center
