# Build instructions for ZRobot ROS2

## On RK3588 (Rock 5B) with Ubuntu 24.04 + ROS2 Jazzy

### 1. Install ROS2 Jazzy

```bash
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install -y ros-jazzy-desktop
source /opt/ros/jazzy/setup.bash
```

### 2. Install dependencies

```bash
sudo apt install -y \
    ros-jazzy-cv-bridge \
    ros-jazzy-vision-msgs \
    ros-jazzy-image-transport \
    ros-jazzy-rknn-toolkit \
    python3-aiohttp \
    python3-numpy \
    python3-opencv \
    python3-pip \
    libyaml-cpp-dev \
    colcon-common-extensions
```

### 3. Copy RKNN model

```bash
mkdir -p models
cp /path/to/yolov8n.rknn models/
```

### 4. Build

```bash
cd ZRobot_ROS
colcon build --symlink-install
source install/setup.bash
```

### 5. Run

```bash
# Full system
ros2 launch zrobot_bringup zrobot_launch.py

# Or individual nodes in separate terminals:
ros2 run zrobot_perception yolo_detector_node
ros2 run zrobot_control motor_controller_node  
ros2 run web_interface web_server
```

### 6. Access web interface

```
http://localhost:8080  # On device
http://<ROBOT_IP>:8080 # From network
```

## Troubleshooting

### RKNN library not found
```bash
# Check library exists
ls -la /usr/lib/librknn_api.so

# Set LD_LIBRARY_PATH if needed
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
```

### Camera permission denied
```bash
sudo usermod -a -G video $USER
# Logout and login again
```

### UART permission denied
```bash
sudo usermod -a -G dialout $USER
# Logout and login again
```
