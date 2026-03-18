#!/usr/bin/env python3
"""
Obstacle Avoidance Node for ZRobot
Uses LD lidar to detect obstacles and maneuver around them while keeping target in view
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String
import math


class ObstacleAvoidanceNode(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance')

        self.declare_parameter('min_safe_distance', 0.3)
        self.declare_parameter('slow_down_distance', 0.5)
        self.declare_parameter('max_linear_speed', 0.3)
        self.declare_parameter('turn_speed', 0.5)
        self.declare_parameter('scan_angle_range', 60.0)

        self.min_safe_distance = self.get_parameter('min_safe_distance').value
        self.slow_down_distance = self.get_parameter('slow_down_distance').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.turn_speed = self.get_parameter('turn_speed').value
        self.scan_angle_range = self.get_parameter('scan_angle_range').value

        self.cmd_vel_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            String,
            'detection_status',
            self.detection_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10
        )

        self.cmd_vel_safe_pub = self.create_publisher(Twist, 'cmd_vel_safe', 10)
        self.obstacle_pub = self.create_publisher(Bool, 'obstacle_detected', 10)
        self.status_pub = self.create_publisher(String, 'obstacle_status', 10)
        self.lidar_pub = self.create_publisher(String, 'lidar_status', 10)

        self.current_cmd = Twist()
        self.target_in_view = False
        self.target_zone = 'NONE'
        self.last_scan_time = self.get_clock().now()
        self.min_lidar_distance = 999.0

        self.get_logger().info('Obstacle Avoidance Node started')

    def cmd_vel_callback(self, msg):
        self.current_cmd = msg

    def detection_callback(self, msg):
        try:
            import json
            data = json.loads(msg.data)
            self.target_in_view = data.get('found', False)
            self.target_zone = data.get('zone', 'NONE')
        except:
            pass

    def scan_callback(self, msg):
        self.last_scan_time = self.get_clock().now()
        
        ranges = msg.ranges
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        
        angle_range_rad = math.radians(self.scan_angle_range)
        
        front_left_ranges = []
        front_right_ranges = []
        front_ranges = []
        
        for i, r in enumerate(ranges):
            if math.isinf(r) or math.isnan(r):
                continue
            
            angle = angle_min + i * angle_increment
            
            if -angle_range_rad/2 < angle < angle_range_rad/2:
                front_ranges.append(r)
            
            if -angle_range_rad < angle < 0:
                front_left_ranges.append(r)
            
            if 0 < angle < angle_range_rad:
                front_right_ranges.append(r)
        
        min_front = min(front_ranges) if front_ranges else float('inf')
        min_front_left = min(front_left_ranges) if front_left_ranges else float('inf')
        min_front_right = min(front_right_ranges) if front_right_ranges else float('inf')

        output_cmd = Twist()
        
        if min_front < self.min_safe_distance:
            output_cmd.linear.x = 0.0
            output_cmd.angular.z = 0.0
            
            if self.target_in_view:
                if self.target_zone == 'LEFT':
                    output_cmd.angular.z = -self.turn_speed
                elif self.target_zone == 'RIGHT':
                    output_cmd.angular.z = self.turn_speed
                elif self.target_zone == 'CENTER':
                    if min_front_left > min_front_right:
                        output_cmd.angular.z = -self.turn_speed
                    else:
                        output_cmd.angular.z = self.turn_speed
                else:
                    output_cmd.angular.z = self.turn_speed
            else:
                if min_front_left > min_front_right:
                    output_cmd.angular.z = -self.turn_speed
                else:
                    output_cmd.angular.z = self.turn_speed
            
            obstacle_msg = Bool()
            obstacle_msg.data = True
            self.obstacle_pub.publish(obstacle_msg)
            
        elif min_front < self.slow_down_distance:
            slow_factor = (min_front - self.min_safe_distance) / (self.slow_down_distance - self.min_safe_distance)
            slow_factor = max(0.2, min(1.0, slow_factor))
            
            output_cmd.linear.x = self.current_cmd.linear.x * slow_factor
            output_cmd.angular.z = self.current_cmd.angular.z
            
            obstacle_msg = Bool()
            obstacle_msg.data = True
            self.obstacle_pub.publish(obstacle_msg)
        else:
            output_cmd = self.current_cmd
            
            obstacle_msg = Bool()
            obstacle_msg.data = False
            self.obstacle_pub.publish(obstacle_msg)

        self.cmd_vel_safe_pub.publish(output_cmd)
        
        # Publish obstacle status with distance
        import json
        status_msg = String()
        status_data = {
            'obstacle': min_front < self.min_safe_distance if min_front != float('inf') else False,
            'distance': min_front if min_front != float('inf') else 999.0
        }
        status_msg.data = json.dumps(status_data)
        self.status_pub.publish(status_msg)
        self.min_lidar_distance = min_front if min_front != float('inf') else 999.0

        # Publish detailed lidar status
        lidar_msg = String()
        lidar_data = {
            'connected': True,
            'fps': 10,
            'points': len(ranges),
            'range_max': msg.range_max if hasattr(msg, 'range_max') else 12.0,
            'front_distance': min_front if min_front != float('inf') else 999.0,
            'left_distance': min_front_left if min_front_left != float('inf') else 999.0,
            'right_distance': min_front_right if min_front_right != float('inf') else 999.0
        }
        lidar_msg.data = json.dumps(lidar_data)
        self.lidar_pub.publish(lidar_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidanceNode()
    
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
