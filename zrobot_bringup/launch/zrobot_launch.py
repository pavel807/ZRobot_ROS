#!/usr/bin/env python3
"""
ZRobot Bringup Launch File
Starts all ZRobot nodes: perception, control, web interface, lidar, and obstacle avoidance
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import SetEnvironmentVariable, UnsetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # --- 1. Пути к файлам ---
    pkg_bringup = FindPackageShare('zrobot_bringup')
    
    config_file = PathJoinSubstitution([
        pkg_bringup,
        'config',
        'zrobot_config.yaml'
    ])

    # --- 2. Аргументы запуска (можно менять через командную строку) ---
    return LaunchDescription([
        # Use FastDDS (default on many Jazzy installs), but explicitly unset the
        # profile env var that points to an incompatible XML (segment_count, etc).
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        UnsetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE'),

        DeclareLaunchArgument(
            'model_path',
            default_value='models/yolov8s.rknn',
            description='Path to YOLO RKNN model'
        ),
        DeclareLaunchArgument(
            'camera_id',
            default_value='1',
            description='Camera device ID'
        ),
        DeclareLaunchArgument(
            'uart_port',
            default_value='/dev/ttyACM0',
            description='UART port for Arduino'
        ),
        DeclareLaunchArgument(
            'baud_rate',
            default_value='115200',
            description='Baud rate for Arduino'
        ),
        DeclareLaunchArgument(
            'conf_threshold',
            default_value='0.45',
            description='Confidence threshold for detection'
        ),
        DeclareLaunchArgument(
            'target_object',
            default_value='person',
            description='Target object to track'
        ),

        # --- 3. Узлы (Nodes) ---

        # YOLO Detector Node
        Node(
            package='zrobot_perception',
            executable='yolo_detector_node',
            name='yolo_detector',
            output='screen',
            parameters=[
                config_file,
                {
                    'model_path': LaunchConfiguration('model_path'),
                    'camera_id': LaunchConfiguration('camera_id'),
                    'conf_threshold': LaunchConfiguration('conf_threshold'),
                    'target_object': LaunchConfiguration('target_object'),
                }
            ]
        ),

        # Motor Controller Node (Arduino)
        Node(
            package='zrobot_control',
            executable='motor_controller_node',
            name='motor_controller',
            output='screen',
            parameters=[
                config_file,
                {
                    'uart_port': LaunchConfiguration('uart_port'),
                    'baud_rate': LaunchConfiguration('baud_rate'),
                }
            ]
        ),

        # Web Interface Node
        Node(
            package='web_interface',
            executable='web_server',
            name='web_interface',
            output='screen',
            parameters=[{
                'host': '0.0.0.0',
                'port': 8080,
            }]
        ),

        # LD Lidar Node (LD19)
        Node(
            package='ldlidar_stl_ros2',
            executable='ldlidar_stl_ros2_node',
            name='ldlidar',
            output='screen',
            respawn=True,
            respawn_delay=2.0,
            parameters=[{
                'product_name': 'LDLiDAR_LD19',
                'topic_name': 'scan',
                'port_name': '/dev/ttyUSB0',
                'port_baudrate': 230400,
                'frame_id': 'base_laser',
                'laser_scan_dir': True,
                'enable_angle_crop_func': False,
            }]
        ),

        # Obstacle Avoidance Node
        Node(
            package='zrobot_obstacle_avoidance',
            executable='obstacle_avoidance_node.py',
            name='obstacle_avoidance',
            output='screen',
            parameters=[{
                'enabled': True,
                'min_safe_distance': 0.3,
                'slow_down_distance': 0.5,
                'max_linear_speed': 0.3,
                'turn_speed': 0.5,
                'scan_angle_range': 60.0,
                'scan_timeout_sec': 1.0,
            }]
        ),
    ])
