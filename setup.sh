#!/bin/bash
# ZRobot ROS2 Setup Script

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# Error tracking
ERRORS=0

log_error() {
    echo -e "${RED}✗ $1${NC}"
    ((ERRORS++))
}

log_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Spinner chars
SPINNER=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
SPIN_LEN=10

# ASCII Art
ZROBOT_ART="
   ██████╗ ███████╗████████╗██████╗  ██████╗ ██████╗  █████╗ ██████╗  ██████╗ 
  ██╔════╝ ██╔════╝╚══██╔══╝██╔══██╗██╔═══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝ 
  ██║  ███╗█████╗     ██║   ██████╔╝██║   ██║██████╔╝███████║██████╔╝██║  ███╗
  ██║   ██║██╔══╝     ██║   ██╔══██╗██║   ██║██╔══██╗██╔══██║██╔══██╗██║   ██║
  ╚██████╔╝███████╗   ██║   ██║  ██║╚██████╔╝██████╔╝██║  ██║██║  ██║╚██████╔╝
   ╚═════╝ ╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ 
"

print_header() {
    clear
    echo -e "${CYAN}${ZROBOT_ART}${NC}"
    echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}                 ZRobot ROS2 Setup${NC}"
    echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

spinner() {
    local pid=$1
    local delay=0.1
    while kill -0 $pid 2>/dev/null; do
        for i in $(seq 0 $((SPIN_LEN-1))); do
            printf "\r${YELLOW}${SPINNER[$i]}${NC} "
            sleep $delay
        done
    done
    printf "\r"
}

check_ros() {
    print_header
    echo -e "${YELLOW}▸ Проверка ROS2...${NC}"
    
    if [ -z "$ROS_DISTRO" ]; then
        echo -e "${RED}✗ ROS2 не найден!${NC}"
        echo ""
        echo -e "${CYAN}Сначала запустите:${NC}"
        echo -e "  ${GREEN}source /opt/ros/jazzy/setup.bash${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ ROS2 $ROS_DISTRO найден${NC}"
    echo ""
}

install_deps() {
    print_header
    echo -e "${YELLOW}▸ Установка зависимостей...${NC}"
    echo ""
    
    # DDS
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    unset FASTRTPS_DEFAULT_PROFILES_FILE
    
    # Python deps
    echo -e "${BLUE}  Python пакеты...${NC}"
    if pip3 install --break-system-packages --quiet 'numpy<2' aiohttp opencv-python-headless psutil 2>/dev/null; then
        echo -e "${GREEN}  ✓ Python зависимости${NC}"
    else
        log_error "Не удалось установить Python пакеты"
    fi
    echo ""
    
    # ROS2 deps
    echo -e "${BLUE}  ROS2 пакеты...${NC}"
    if sudo apt-get update -qq 2>/dev/null; then
        if sudo apt-get install -y -qq \
            ros-$ROS_DISTRO-cv-bridge \
            ros-$ROS_DISTRO-vision-msgs \
            ros-$ROS_DISTRO-image-transport \
            ros-$ROS_DISTRO-libopencv-dev \
            ros-$ROS_DISTRO-sensor-msgs \
            ros-$ROS_DISTRO-geometry-msgs \
            ros-$ROS_DISTRO-std-msgs \
            ros-$ROS_DISTRO-rclcpp \
            ros-$ROS_DISTRO-rclpy \
            ros-$ROS_DISTRO-message-filters \
            python3-aiohttp \
            python3-numpy \
            python3-opencv \
            python3-psutil \
            libyaml-cpp-dev \
            2>/dev/null; then
            echo -e "${GREEN}  ✓ ROS2 зависимости${NC}"
        else
            log_error "Не удалось установить ROS2 пакеты"
        fi
    else
        log_error "Не удалось обновить apt"
    fi
    echo ""
}

setup_models() {
    print_header
    echo -e "${YELLOW}▸ Настройка директорий...${NC}"
    echo ""
    
    mkdir -p models
    echo -e "${GREEN}✓ Директория models создана${NC}"
    
    if [ ! -f "zrobot_config.yaml" ]; then
        echo -e "${YELLOW}⚠ Конфиг не найден! Создаю по умолчанию...${NC}"
        cat > zrobot_config.yaml << 'EOF'
# ZRobot Configuration

zrobot_perception:
  model_path: "models/yolo26s-rk3588.rknn"
  camera_id: 1
  conf_threshold: 0.30
  target_object: "person"
  max_linear_speed: 0.3
  turn_speed: 0.5

zrobot_control:
  uart_port: "/dev/ttyACM0"
  baud_rate: 115200
  max_speed: 245
  min_speed: 165

zrobot_lidar:
  port_name: "/dev/ttyUSB0"
  port_baudrate: 230400

zrobot_obstacle_avoidance:
  enabled: true
  min_safe_distance: 0.3

zrobot_web:
  port: 8080
EOF
    else
        echo -e "${GREEN}✓ Конфиг zrobot_config.yaml найден${NC}"
    fi
    
    if [ ! -f "models/yolo26s-rk3588.rknn" ]; then
        echo -e "${YELLOW}⚠ Модель не найдена! Поместите yolo26s-rk3588.rknn в папку models/${NC}"
    else
        echo -e "${GREEN}✓ Модель yolo26s-rk3588.rknn найдена${NC}"
    fi
    echo ""
}

build_workspace() {
    print_header
    echo -e "${YELLOW}▸ Сборка workspace...${NC}"
    echo ""
    
    # Check for packages (either in src/ or root)
    if [ ! -d "src" ] && [ -z "$(ls -d zrobot_* 2>/dev/null)" ]; then
        echo -e "${RED}✗ Пакеты не найдены!${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}  Компиляция...${NC}"
    if colcon build --symlink-install 2>&1 | tail -20; then
        echo ""
        echo -e "${GREEN}✓ Сборка завершена${NC}"
    else
        log_error "Ошибка сборки"
    fi
    echo ""
}

show_summary() {
    print_header
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}                     ✨ Установка завершена!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    if [ $ERRORS -gt 0 ]; then
        echo -e "${YELLOW}⚠ Было ошибок: $ERRORS${NC}"
        echo ""
    fi
    
    echo -e "${CYAN}Для запуска:${NC}"
    echo -e "  ${YELLOW}source install/setup.bash${NC}"
    echo -e "  ${YELLOW}ros2 launch zrobot_bringup zrobot_launch.py${NC}"
    echo ""
    echo -e "${CYAN}Веб-интерфейс:${NC}"
    echo -e "  ${YELLOW}http://localhost:8080${NC}"
    echo ""
    echo -e "${CYAN}Конфиг:${NC}"
    echo -e "  ${GREEN}zrobot_config.yaml${NC} (редактируй - пересборка не нужна)"
    echo ""
}

main() {
    check_ros
    install_deps
    setup_models
    build_workspace
    show_summary
}

main "$@"
