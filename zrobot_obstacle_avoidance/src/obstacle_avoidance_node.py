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
import time
from collections import deque


class ObstacleAvoidanceNode(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance')

        self.declare_parameter('enabled', True)
        self.declare_parameter('min_safe_distance', 0.3)
        self.declare_parameter('slow_down_distance', 0.5)
        self.declare_parameter('max_linear_speed', 0.3)
        self.declare_parameter('turn_speed', 0.5)
        self.declare_parameter('scan_angle_range', 60.0)
        self.declare_parameter('front_offset', 0.205)
        self.declare_parameter('scan_timeout_sec', 1.0)

        self.enabled = bool(self.get_parameter('enabled').value)
        self.min_safe_distance = self.get_parameter('min_safe_distance').value
        self.slow_down_distance = self.get_parameter('slow_down_distance').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.turn_speed = self.get_parameter('turn_speed').value
        self.scan_angle_range = float(self.get_parameter('scan_angle_range').value)
        self.front_offset = self.get_parameter('front_offset').value
        self.scan_timeout_sec = float(self.get_parameter('scan_timeout_sec').value)

        self.cmd_vel_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )

        self.obstacle_enable_sub = self.create_subscription(
            Bool,
            'obstacle_enable',
            self.obstacle_enable_callback,
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
        self._scan_stamps = deque(maxlen=30)
        self._last_scan_points = 0
        self._last_range_max = 0.0
        self._last_sector_distances = {
            'front': 999.0,
            'rear': 999.0,
            'left': 999.0,
            'right': 999.0,
        }

        self._status_timer = self.create_timer(0.5, self._publish_periodic_status)

        self.get_logger().info('Obstacle Avoidance Node started')

    def cmd_vel_callback(self, msg):
        self.current_cmd = msg

    def obstacle_enable_callback(self, msg: Bool):
        self.enabled = bool(msg.data)

    def detection_callback(self, msg):
        try:
            import json
            data = json.loads(msg.data)
            self.target_in_view = data.get('found', False)
            self.target_zone = data.get('zone', 'NONE')
        except:
            pass

    @staticmethod
    def _angle_deg_0_360(angle_rad: float) -> float:
        deg = math.degrees(angle_rad) % 360.0
        if deg < 0:
            deg += 360.0
        return deg

    @staticmethod
    def _within_sector(deg: float, center_deg: float, half_width_deg: float) -> bool:
        # shortest signed distance on a circle [-180, 180]
        delta = (deg - center_deg + 180.0) % 360.0 - 180.0
        return abs(delta) <= half_width_deg

    def _clamp_cmd(self, cmd: Twist) -> Twist:
        out = Twist()
        out.linear.x = max(-self.max_linear_speed, min(self.max_linear_speed, float(cmd.linear.x)))
        out.angular.z = max(-abs(self.turn_speed), min(abs(self.turn_speed), float(cmd.angular.z)))
        return out

    def _publish_periodic_status(self):
        # Emit lidar_status even when scan is missing, so UI can show disconnected state.
        now = self.get_clock().now()
        dt = (now - self.last_scan_time).nanoseconds / 1e9
        connected = dt <= self.scan_timeout_sec

        import json
        lidar_msg = String()
        lidar_data = {
            'connected': bool(connected),
            'fps': self._estimate_scan_fps(),
            'points': int(self._last_scan_points),
            'range_max': float(self._last_range_max),
            'front_distance': float(self._last_sector_distances['front']),
            'rear_distance': float(self._last_sector_distances['rear']),
            'left_distance': float(self._last_sector_distances['left']),
            'right_distance': float(self._last_sector_distances['right']),
            'error': None if connected else 'scan_timeout',
        }
        lidar_msg.data = json.dumps(lidar_data)
        self.lidar_pub.publish(lidar_msg)

    def _estimate_scan_fps(self) -> float:
        if len(self._scan_stamps) < 2:
            return 0.0
        duration = self._scan_stamps[-1] - self._scan_stamps[0]
        if duration <= 0:
            return 0.0
        return round((len(self._scan_stamps) - 1) / duration, 1)

    def scan_callback(self, msg):
        self.last_scan_time = self.get_clock().now()
        self._scan_stamps.append(time.time())
        self._last_scan_points = len(msg.ranges)
        self._last_range_max = float(getattr(msg, 'range_max', 0.0) or 0.0)
        
        ranges = msg.ranges
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        
        front_ranges = []
        rear_ranges = []
        left_ranges = []
        right_ranges = []

        half_front = max(1.0, self.scan_angle_range / 2.0)
        half_side = max(5.0, min(90.0, self.scan_angle_range))

        for i, r in enumerate(ranges):
            if math.isinf(r) or math.isnan(r) or r <= 0.0:
                continue

            if hasattr(msg, 'range_min') and r < msg.range_min:
                continue
            if hasattr(msg, 'range_max') and msg.range_max > 0.0 and r > msg.range_max:
                continue
            
            angle = angle_min + i * angle_increment
            deg = self._angle_deg_0_360(angle)

            # Convert from lidar measurement to approximate distance from robot front
            dist = max(0.0, float(r) - float(self.front_offset))

            if self._within_sector(deg, center_deg=0.0, half_width_deg=half_front):
                front_ranges.append(dist)
            if self._within_sector(deg, center_deg=180.0, half_width_deg=half_front):
                rear_ranges.append(dist)
            if self._within_sector(deg, center_deg=90.0, half_width_deg=half_side):
                left_ranges.append(dist)
            if self._within_sector(deg, center_deg=270.0, half_width_deg=half_side):
                right_ranges.append(dist)

        min_front = min(front_ranges) if front_ranges else float('inf')
        min_rear = min(rear_ranges) if rear_ranges else float('inf')
        min_left = min(left_ranges) if left_ranges else float('inf')
        min_right = min(right_ranges) if right_ranges else float('inf')

        self._last_sector_distances['front'] = min_front if min_front != float('inf') else 999.0
        self._last_sector_distances['rear'] = min_rear if min_rear != float('inf') else 999.0
        self._last_sector_distances['left'] = min_left if min_left != float('inf') else 999.0
        self._last_sector_distances['right'] = min_right if min_right != float('inf') else 999.0

        # If disabled, pass-through without modification.
        if not self.enabled:
            passthrough = self._clamp_cmd(self.current_cmd)
            self.cmd_vel_safe_pub.publish(passthrough)
            self.obstacle_pub.publish(Bool(data=False))
            self._publish_obstacle_status(min_front, min_rear, min_left, min_right)
            self._publish_lidar_status(msg, min_front, min_rear, min_left, min_right)
            return

        output_cmd = Twist()
        
        if min_front < self.min_safe_distance:
            output_cmd.linear.x = 0.0
            output_cmd.angular.z = 0.0
            
            if self.target_in_view:
                if self.target_zone == 'LEFT':
                    output_cmd.angular.z = self.turn_speed
                elif self.target_zone == 'RIGHT':
                    output_cmd.angular.z = -self.turn_speed
                elif self.target_zone == 'CENTER':
                    if min_left > min_right:
                        output_cmd.angular.z = self.turn_speed
                    else:
                        output_cmd.angular.z = -self.turn_speed
                else:
                    output_cmd.angular.z = -self.turn_speed
            else:
                if min_left > min_right:
                    output_cmd.angular.z = self.turn_speed
                else:
                    output_cmd.angular.z = -self.turn_speed
            
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

        output_cmd = self._clamp_cmd(output_cmd)
        self.cmd_vel_safe_pub.publish(output_cmd)

        self._publish_obstacle_status(min_front, min_rear, min_left, min_right)
        self.min_lidar_distance = min_front if min_front != float('inf') else 999.0

        self._publish_lidar_status(msg, min_front, min_rear, min_left, min_right)

    def _publish_obstacle_status(self, min_front, min_rear, min_left, min_right):
        import json
        status_msg = String()
        status_data = {
            'enabled': bool(self.enabled),
            'obstacle': (min_front < self.min_safe_distance) if min_front != float('inf') else False,
            'distance': min_front if min_front != float('inf') else 999.0,
            'front': min_front if min_front != float('inf') else 999.0,
            'rear': min_rear if min_rear != float('inf') else 999.0,
            'left': min_left if min_left != float('inf') else 999.0,
            'right': min_right if min_right != float('inf') else 999.0
        }
        status_msg.data = json.dumps(status_data)
        self.status_pub.publish(status_msg)

    def _publish_lidar_status(self, scan_msg: LaserScan, min_front, min_rear, min_left, min_right):
        import json
        lidar_msg = String()
        lidar_data = {
            'connected': True,
            'fps': self._estimate_scan_fps(),
            'points': len(scan_msg.ranges),
            'range_max': float(getattr(scan_msg, 'range_max', 0.0) or 0.0),
            'front_distance': min_front if min_front != float('inf') else 999.0,
            'rear_distance': min_rear if min_rear != float('inf') else 999.0,
            'left_distance': min_left if min_left != float('inf') else 999.0,
            'right_distance': min_right if min_right != float('inf') else 999.0,
            'error': None,
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
