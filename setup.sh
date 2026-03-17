#!/bin/bash
# ZRobot ROS2 Setup Script

set -e

echo "=========================================="
echo "ZRobot ROS2 Setup"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check ROS2
if [ -z "$ROS_DISTRO" ]; then
    echo -e "${RED}Error: ROS2 not sourced!${NC}"
    echo "Run: source /opt/ros/jazzy/setup.bash"
    exit 1
fi

echo -e "${GREEN}✓ ROS2 $ROS_DISTRO detected${NC}"

# Install Python dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip3 install aiohttp numpy opencv-python-headless 2>/dev/null || true

# Install ROS2 dependencies
echo -e "${YELLOW}Checking ROS2 dependencies...${NC}"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    ros-$ROS_DISTRO-cv-bridge \
    ros-$ROS_DISTRO-vision-msgs \
    ros-$ROS_DISTRO-image-transport \
    ros-$ROS_DISTRO-libopencv-dev \
    python3-aiohttp \
    python3-numpy \
    python3-opencv \
    libyaml-cpp-dev \
    2>/dev/null || true

# Create models directory
echo -e "${YELLOW}Creating models directory...${NC}"
mkdir -p models
echo "Place your .rknn model files here"

# Build
echo -e "${YELLOW}Building...${NC}"
colcon build --symlink-install

# Source
echo -e "${YELLOW}Sourcing workspace...${NC}"
source install/setup.bash

echo ""
echo -e "${GREEN}=========================================="
echo "Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "To run ZRobot:"
echo "  ros2 launch zrobot_bringup zrobot_launch.py"
echo ""
echo "Web interface: http://localhost:8080"
echo ""
