# ZRobot ROS2

ROS2-based object detection and tracking robot system for RK3588 (Rock 5B) with YOLO and NPU acceleration.

## 📋 Overview

This is the ROS2 migration of the original ZRobot project. It provides:

- **YOLO Object Detection** - Running on RK3588 NPU via RKNN
- **Motor Control** - UART communication with Arduino
- **Web Interface** - Real-time control via browser with WebSocket
- **ROS2 Integration** - Standard messages and nodes for easy integration

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Camera Node    │────▶│ YOLO Detector    │────▶│ Object Tracker  │
│  (V4L2)         │     │ (RKNN NPU)       │     │                 │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
        ┌─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Motor Driver   │◀────│ Motor Controller │◀────│ Web Interface   │
│  (UART/Arduino) │     │                  │     │ (HTTP/WS)       │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## 📦 Packages

| Package | Description |
|---------|-------------|
| `zrobot_perception` | YOLO detection with RKNN |
| `zrobot_control` | Motor control via UART |
| `zrobot_interfaces` | Custom ROS2 messages/services |
| `zrobot_bringup` | Launch files and configuration |
| `web_interface` | Web server with WebSocket |

## 🚀 Quick Start

### Prerequisites

```bash
# ROS2 Jazzy (Ubuntu 24.04)
source /opt/ros/jazzy/setup.bash

# Dependencies
sudo apt-get update
sudo apt-get install -y \
    ros-jazzy-cv-bridge \
    ros-jazzy-vision-msgs \
    ros-jazzy-image-transport \
    python3-aiohttp \
    python3-numpy \
    python3-opencv \
    libyaml-cpp-dev
```

### Build

```bash
cd ~/ZRobot_ROS
colcon build --symlink-install
source install/setup.bash
```

### Run

```bash
# Full system
ros2 launch zrobot_bringup zrobot_launch.py

# Individual nodes
ros2 run zrobot_perception yolo_detector_node
ros2 run zrobot_control motor_controller_node
ros2 run web_interface web_server
```

### Access Web Interface

Open browser: `http://<robot-ip>:8080`

## 🔧 Configuration

Edit `zrobot_bringup/config/zrobot_config.yaml`:

```yaml
zrobot_perception:
  model_path: "models/yolov8n.rknn"
  camera_id: 0
  conf_threshold: 0.3
  
zrobot_control:
  uart_port: "/dev/ttyACM0"
  baud_rate: 9600
  max_speed: 245
```

## 📡 ROS2 Topics

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/detections` | `vision_msgs/Detection2DArray` | Object detections |
| `/processed_image` | `sensor_msgs/Image` | Annotated camera feed |
| `/detection_status` | `std_msgs/String` | JSON status updates |
| `/motor_status` | `std_msgs/String` | Motor status |

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/cmd_vel` | `geometry_msgs/Twist` | Velocity commands |
| `/set_target` | `std_msgs/String` | Target object class |
| `/motor_enable` | `std_msgs/Bool` | Enable/disable motors |

## 🔌 Services

| Service | Type | Description |
|---------|------|-------------|
| `/set_target` | `zrobot_interfaces/SetTarget` | Set tracking target |
| `/set_config` | `zrobot_interfaces/SetConfig` | Update parameters |

## 🛠️ Development

### Add new object classes

Edit COCO_NAMES in `yolo_detector_node.cpp`

### Tune motor parameters

```bash
ros2 param set /motor_controller max_speed 200
ros2 param set /motor_controller min_speed 150
```

### Debug with ROS2 tools

```bash
# View topics
ros2 topic list

# Monitor detections
ros2 topic echo /detections

# Check node status
ros2 node list

# RViz visualization
rviz2
```

## 📝 Notes

- **RKNN Library**: Requires `librknn_api.so` in `/usr/lib`
- **Camera**: V4L2 compatible camera required
- **UART**: Ensure user has dialout group access: `sudo usermod -a -G dialout $USER`

## 📄 License

MIT License

## 🙏 Credits

Based on the original ZRobot project with ROS2 architecture.
