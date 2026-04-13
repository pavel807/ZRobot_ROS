# ZRobot ROS2

Робот на базе ROS2 + YOLO детекция на NPU RK3588 (Rock 5B).

## 📦 Что умеет

- **Детекция объектов** — YOLO на NPU через RKNN
- **Управление моторами** — UART связь с Arduino
- **Веб-интерфейс** — управление через браузер (HTTP + WebSocket)
- **Обход препятствий** — на основе лидара LD LiDAR

## 🚀 Быстрый старт

### 1. Зависимости

```bash
sudo apt update
sudo apt-get install -y -qq \
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
```

### 2. Сборка

```bash
cd ~/ws_ros2/ZRobot_ROS
colcon build --symlink-install
source install/setup.bash
```

### 3. Запуск

```bash
ros2 launch zrobot_bringup zrobot_launch.py
```

Веб-интерфейс: `http://<IP-робота>:8080`

## 📂 Пакеты

| Пакет | Описание |
|-------|----------|
| `zrobot_perception` | YOLO детекция (RKNN NPU) |
| `zrobot_control` | Управление моторами (UART/Arduino) |
| `zrobot_obstacle_avoidance` | Обход препятствий (LiDAR) |
| `zrobot_interfaces` | Кастомные сообщения/сервисы |
| `zrobot_bringup` | Лаунч-файлы и конфиги |
| `web_interface` | Веб-сервер с WebSocket |
| `ldlidar_stl_ros2` | Драйвер LD LiDAR |

## ⚙️ Конфигурация

Править `zrobot_bringup/config/zrobot_config.yaml`:

```yaml
zrobot_perception:
  model_path: "models/yolo26s-rk3588.rknn"  # путь к модели
  camera_id: 1                       # ID камеры
  conf_threshold: 0.30               # порог уверенности

zrobot_control:
  uart_port: "/dev/ttyACM0"
  baud_rate: 115200
```

## 🛠 Решение проблем

| Проблема | Решение |
|----------|---------|
| `librknn_api.so not found` | `ls -la /usr/lib/librknn_api.so` |
| Camera permission denied | `sudo usermod -a -G video $USER` |
| UART permission denied | `sudo usermod -a -G dialout $USER` |
| Port 8080 busy | `sudo fuser -k 8080/tcp` |

## 📄 Лицензия

MIT
