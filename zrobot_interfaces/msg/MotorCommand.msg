# Motor command message
# Published by object_tracker, consumed by motor_controller

int32 left_speed              # Left motor speed (0-255)
int32 right_speed             # Right motor speed (0-255)
bool motors_running           # True if motors are active
string mode                   # "TRACKING", "SEARCHING", "STOPPED"
