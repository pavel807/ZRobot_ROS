#!/usr/bin/env python3
"""
ZRobot Bringup Launch File
Starts all ZRobot nodes: perception, control, and web interface
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='models/yolov8s.rknn',
        description='Path to YOLO RKNN model'
    )

    camera_id_arg = DeclareLaunchArgument(
        'camera_id',
        default_value='1',
        description='Camera device ID'
    )

    uart_port_arg = DeclareLaunchArgument(
        'uart_port',
        default_value='/dev/ttyACM0',
        description='UART port for Arduino'
    )

    conf_threshold_arg = DeclareLaunchArgument(
        'conf_threshold',
        default_value='0.45',
        description='Confidence threshold for detection'
    )

    target_object_arg = DeclareLaunchArgument(
        'target_object',
        default_value='person',
        description='Target object to track'
    )

    # Load config file
    config_file = PathJoinSubstitution([
        FindPackageShare('zrobot_bringup'),
        'config',
        'zrobot_config.yaml'
    ])

    # YOLO Detector Node
    yolo_detector_node = Node(
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
                'adaptive_threshold': False,
            }
        ]
    )

    # Motor Controller Node
    motor_controller_node = Node(
        package='zrobot_control',
        executable='motor_controller_node',
        name='motor_controller',
        output='screen',
        parameters=[
            config_file,
            {
                'uart_port': LaunchConfiguration('uart_port'),
            }
        ]
    )

    # Web Interface Node (Python)
    web_interface_node = Node(
        package='web_interface',
        executable='web_server',
        name='web_interface',
        output='screen',
        parameters=[{
            'host': '0.0.0.0',
            'port': 8080,
        }]
    )

    # LD Lidar Node
    lidar_node = Node(
        package='ldlidar_stl_ros2',
        executable='ldlidar_stl_ros2_node',
        name='ldlidar',
        output='screen',
        parameters=[{
            'serial_port': '/dev/ttyUSB0',
            'serial_baudrate': 115200,
            'frame_id': 'laser_frame',
            'inverted': False,
            'angle_min': -3.14159,
            'angle_max': 3.14159,
            'range_min': 0.1,
            'range_max': 30.0,
            'scan_frequency': 10,
        }]
    )

    # Obstacle Avoidance Node
    obstacle_avoidance_node = Node(
        package='zrobot_obstacle_avoidance',
        executable='obstacle_avoidance_node.py',
        name='obstacle_avoidance',
        output='screen',
        parameters=[{
            'min_safe_distance': 0.3,
            'slow_down_distance': 0.5,
            'max_linear_speed': 0.3,
            'turn_speed': 0.5,
            'scan_angle_range': 60.0,
        }]
    )

    return LaunchDescription([
        model_path_arg,
        camera_id_arg,
        uart_port_arg,
        conf_threshold_arg,
        target_object_arg,
        yolo_detector_node,
        motor_controller_node,
        web_interface_node,
        lidar_node,
        obstacle_avoidance_node,
    ])
