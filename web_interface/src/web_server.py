#!/usr/bin/env python3
"""
ZRobot Web Interface Node - Advanced Modern UI
HTTP server with WebSocket for real-time robot control and settings
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool, Float32
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import json
import threading
import asyncio
import time
import psutil
import os
from collections import deque
from aiohttp import web
import base64
from cv_bridge import CvBridge
import argparse


class WebInterfaceNode(Node):
    def __init__(self):
        super().__init__('web_interface')

        self.declare_parameter('host', '0.0.0.0')
        self.declare_parameter('port', 8080)
        self.declare_parameter('state_update_rate', 5.0)

        self.host = self.get_parameter('host').get_parameter_value().string_value
        self.port = self.get_parameter('port').get_parameter_value().integer_value
        self.state_update_rate = self.get_parameter('state_update_rate').get_parameter_value().double_value

        self.start_time = time.time()

        self.image_sub = self.create_subscription(
            Image, 'processed_image', self.image_callback, 10)
        self.status_sub = self.create_subscription(
            String, 'detection_status', self.status_callback, 10)
        self.detections_sub = self.create_subscription(
            Detection2DArray, 'detections', self.detections_callback, 10)
        self.tracked_sub = self.create_subscription(
            String, 'tracked_objects', self.tracked_callback, 10)
        self.obstacle_sub = self.create_subscription(
            Bool, 'obstacle_detected', self.obstacle_callback, 10)
        self.motor_status_sub = self.create_subscription(
            String, 'motor_status', self.motor_status_callback, 10)
        self.obstacle_status_sub = self.create_subscription(
            String, 'obstacle_status', self.obstacle_status_callback, 10)
        self.lidar_sub = self.create_subscription(
            String, 'lidar_status', self.lidar_callback, 10)
        self.lidar_scan_sub = self.create_subscription(
            String, 'lidar_scan', self.lidar_scan_callback, 10)

        self.target_pub = self.create_publisher(String, 'set_target', 10)
        self.confidence_pub = self.create_publisher(Float32, 'set_confidence', 10)
        self.motor_enable_pub = self.create_publisher(Bool, 'motor_enable', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.obstacle_enable_pub = self.create_publisher(Bool, 'obstacle_enable', 10)
        self.config_pub = self.create_publisher(String, 'set_config', 10)

        self.bridge = CvBridge()
        self.frame_lock = threading.Lock()
        self.current_frame = None
        self.websocket_clients = set()

        self.log_buffer = deque(maxlen=100)
        self.alerts = deque(maxlen=20)

        self.detection_history = deque(maxlen=30)
        self.motor_history = deque(maxlen=30)

        # Alerts/notifications disabled (they can cause UI lag on some devices)
        self.last_obstacle_alert_time = 0
        self.obstacle_alert_cooldown = 0.0

        self._init_system_state()
        self._init_settings()

        self.get_logger().info('Web Interface Node initialized')

    def _init_system_state(self):
        self.system_state = {
            'video': {
                'fps': 0,
                'resolution': '640x480',
                'codec': 'mjpeg',
                'frame_count': 0,
                'bytes_sent': 0
            },
            'detection': {
                'target': '-',
                'found': False,
                'zone': 'NONE',
                'confidence': 0.0,
                'detections_count': 0,
                'tracked_count': 0,
                'inference_time_ms': 0,
                'fps': 0,
                'classes_detected': [],
                'last_detection_time': None
            },
            'motor': {
                'enabled': True,
                'speed': 0,
                'max_speed': 245,
                'current_command': {'linear': 0.0, 'angular': 0.0},
                'battery_voltage': 0.0,
                'status': 'stopped',
                'error': None
            },
            'obstacle_avoidance': {
                'enabled': True,
                'obstacle_detected': False,
                'min_distance_m': 999.0,
                'left_distance': 999.0,
                'right_distance': 999.0,
                'front_distance': 999.0,
                'status': 'initializing',
                'min_safe_distance': 0.3,
                'slow_down_distance': 0.5
            },
            'lidar': {
                'connected': False,
                'fps': 0,
                'range_max': 0.0,
                'points': 0,
                'error': None,
                'scan_data': None
            },
            'camera': {
                'id': 1,
                'name': 'USB Camera',
                'resolution': '640x480',
                'fps': 30,
                'connected': True
            },
            'system': {
                'uptime_seconds': 0,
                'cpu_usage': 0.0,
                'memory_usage_mb': 0,
                'memory_percent': 0.0,
                'temperature': 0.0,
                'ros2_nodes': [],
                'node_status': {}
            },
            'settings': {},
            'logs': [],
            'alerts': []
        }

    def _init_settings(self):
        self.settings = {
            'target_object': 'person',
            'conf_threshold': 0.45,
            'camera_id': 1,
            'enable_tracking': True,
            'adaptive_threshold': False,
            'show_category': True,
            'motor_enabled': True,
            'max_speed': 200,
            'obstacle_avoidance_enabled': True,
            'min_safe_distance': 0.3,
            'slow_down_distance': 0.5,
            'turn_speed': 0.5,
            'max_linear_speed': 0.3
        }
        self.system_state['settings'] = self.settings.copy()

    def image_callback(self, msg):
        with self.frame_lock:
            try:
                self.current_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                self.system_state['video']['frame_count'] += 1
            except Exception as e:
                self._add_log('ERROR', f'Image conversion error: {e}')

    def status_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.system_state['detection'].update({
                'target': data.get('target', '-'),
                'found': data.get('found', False),
                'zone': data.get('zone', 'NONE'),
                'inference_time_ms': data.get('inference_time', 0),
                'fps': data.get('fps', 0),
                'detections_count': data.get('count', 0),
                'tracked_count': data.get('tracked', 0),
                'classes_detected': data.get('classes', '').split(', ') if data.get('classes') else [],
                'last_detection_time': time.time()
            })
            self.detection_history.append({
                'timestamp': time.time(),
                'found': data.get('found', False),
                'inference_time': data.get('inference_time', 0)
            })
        except Exception as e:
            self._add_log('WARN', f'Status parse error: {e}')

    def detections_callback(self, msg):
        self.system_state['detection']['detections_count'] = len(msg.detections)
        objects_list = []
        if msg.detections:
            scores = []
            for det in msg.detections:
                if det.results:
                    hyp = det.results[0].hypothesis
                    scores.append(hyp.score)
                    bbox = det.bbox
                    objects_list.append({
                        'class': hyp.class_id,
                        'confidence': round(hyp.score, 3),
                        'x': round(bbox.center.position.x - bbox.size_x / 2, 1),
                        'y': round(bbox.center.position.y - bbox.size_y / 2, 1),
                        'w': round(bbox.size_x, 1),
                        'h': round(bbox.size_y, 1),
                        'cx': round(bbox.center.position.x, 1),
                        'cy': round(bbox.center.position.y, 1)
                    })
            if scores:
                self.system_state['detection']['confidence'] = max(scores)
        self.system_state['detection']['objects'] = objects_list

    def tracked_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.system_state['detection']['tracked_count'] = data.get('tracked', 0)
        except:
            pass

    def obstacle_callback(self, msg):
        self.system_state['obstacle_avoidance']['obstacle_detected'] = msg.data
        # notifications disabled

    def obstacle_status_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.system_state['obstacle_avoidance'].update({
                'obstacle_detected': data.get('obstacle', False),
                'min_distance_m': data.get('distance', 999.0),
                'status': 'blocked' if data.get('obstacle') else 'clear'
            })
        except:
            pass

    def motor_status_callback(self, msg):
        try:
            data = json.loads(msg.data)
            left = int(data.get('left_speed', 0))
            right = int(data.get('right_speed', 0))
            speed = max(abs(left), abs(right))

            arduino_event = data.get('arduino_last_event') or data.get('arduino_last_line')
            error = arduino_event if isinstance(arduino_event, str) and ('EMERGENCY' in arduino_event or 'TIMEOUT' in arduino_event) else None

            self.system_state['motor'].update({
                'enabled': bool(data.get('enabled', True)),
                'connected': bool(data.get('connected', False)),
                'left_speed': left,
                'right_speed': right,
                'speed': speed,
                'last_cmd_age_ms': data.get('last_cmd_age_ms', None),
                'last_rx_age_ms': data.get('last_rx_age_ms', None),
                'arduino_seen': bool(data.get('arduino_seen', False)),
                'arduino_last_line': data.get('arduino_last_line', None),
                'arduino_last_event': data.get('arduino_last_event', None),
                'status': 'running' if (speed > 0 and bool(data.get('enabled', True))) else 'stopped',
                'error': error,
            })
            self.motor_history.append({
                'timestamp': time.time(),
                'speed': speed,
                'enabled': bool(data.get('enabled', True))
            })
        except Exception as e:
            self._add_log('WARN', f'Motor status parse error: {e}')

    def lidar_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.system_state['lidar'].update({
                'connected': data.get('connected', False),
                'fps': data.get('fps', 0),
                'points': data.get('points', 0),
                'error': data.get('error')
            })
        except:
            pass

    def lidar_scan_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.system_state['lidar']['scan_data'] = data
        except:
            pass

    def _add_log(self, level, message):
        entry = {
            'level': level,
            'message': message,
            'timestamp': int(time.time() * 1000)
        }
        self.log_buffer.append(entry)
        if len(self.log_buffer) > 100:
            self.log_buffer.popleft()

    def _add_alert(self, level, message):
        # notifications disabled
        return

    def update_system_stats(self):
        uptime = time.time() - self.start_time
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            temp = 0.0
            try:
                temp = psutil.sensors_temperatures().get('cpu_thermal', [{}])[0].get('current', 0.0)
            except:
                pass

            self.system_state['system'].update({
                'uptime_seconds': int(uptime),
                'cpu_usage': round(cpu, 1),
                'memory_usage_mb': int(mem.used / 1024 / 1024),
                'memory_percent': round(mem.percent, 1),
                'temperature': round(temp, 1)
            })
        except:
            pass

        self.system_state['system']['uptime_seconds'] = int(uptime)

        self.system_state['logs'] = list(self.log_buffer)[-20:]
        self.system_state['alerts'] = list(self.alerts)[-10:]

    async def broadcast_state(self):
        if not self.websocket_clients:
            return

        self.update_system_stats()

        message = {
            'type': 'system_state',
            'timestamp': int(time.time() * 1000),
            'data': self.system_state
        }

        await self._send_to_all(json.dumps(message))

    async def _send_to_all(self, message):
        if self.websocket_clients:
            await asyncio.gather(
                *[client.send_str(message) for client in self.websocket_clients],
                return_exceptions=True
            )

    def run_server(self):
        app = web.Application()
        app.router.add_get('/', self.handle_index)
        app.router.add_get('/api/frame', self.handle_frame)
        app.router.add_get('/api/status', self.handle_status)
        app.router.add_get('/api/state', self.handle_full_state)
        app.router.add_get('/api/logs', self.handle_logs)
        app.router.add_post('/api/settings', self.handle_settings)
        app.router.add_post('/api/target', self.handle_set_target)
        app.router.add_post('/api/confidence', self.handle_set_confidence)
        app.router.add_post('/api/motor', self.handle_motor_control)
        app.router.add_post('/api/speed', self.handle_set_speed)
        app.router.add_post('/api/obstacle', self.handle_obstacle_config)
        app.router.add_get('/ws', self.handle_websocket)

        self.broadcast_task = threading.Thread(target=self._broadcast_loop, daemon=True)
        self.broadcast_task.start()

        web.run_app(app, host=self.host, port=self.port, handle_signals=False)

    def _broadcast_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        interval = 1.0 / self.state_update_rate
        while True:
            try:
                loop.run_until_complete(self.broadcast_state())
            except Exception as e:
                pass
            time.sleep(interval)

    async def handle_index(self, request):
        return web.Response(text=self._get_html(), content_type='text/html')

    async def handle_frame(self, request):
        with self.frame_lock:
            if self.current_frame is None:
                return web.Response(status=204)
            _, buffer = cv2.imencode('.jpg', self.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            self.system_state['video']['bytes_sent'] += len(buffer)
            return web.Response(body=buffer.tobytes(), content_type='image/jpeg')

    async def handle_status(self, request):
        return web.json_response(self.system_state['detection'])

    async def handle_full_state(self, request):
        self.update_system_stats()
        return web.json_response(self.system_state)

    async def handle_logs(self, request):
        limit = int(request.query.get('limit', 50))
        return web.json_response({
            'logs': list(self.log_buffer)[-limit:],
            'alerts': list(self.alerts)[-20:]
        })

    async def handle_settings(self, request):
        data = await request.json()
        for key, value in data.items():
            if key in self.settings:
                self.settings[key] = value
                self.system_state['settings'][key] = value
                self._add_log('INFO', f'Setting updated: {key} = {value}')

                if key == 'target_object':
                    msg = String()
                    msg.data = str(value)
                    self.target_pub.publish(msg)
                elif key == 'obstacle_avoidance_enabled':
                    msg = Bool()
                    msg.data = bool(value)
                    self.obstacle_enable_pub.publish(msg)
                    self.system_state['obstacle_avoidance']['enabled'] = bool(value)
                elif key == 'min_safe_distance':
                    self.system_state['obstacle_avoidance']['min_safe_distance'] = float(value)
                elif key == 'slow_down_distance':
                    self.system_state['obstacle_avoidance']['slow_down_distance'] = float(value)

        config_msg = String()
        config_msg.data = json.dumps(self.settings)
        self.config_pub.publish(config_msg)

        return web.json_response({'success': True, 'settings': self.settings})

    async def handle_set_target(self, request):
        data = await request.json()
        target = data.get('target', 'person')
        msg = String()
        msg.data = target
        self.target_pub.publish(msg)
        self.settings['target_object'] = target
        self.system_state['settings']['target_object'] = target
        self._add_log('INFO', f'Target changed to: {target}')
        return web.json_response({'success': True, 'target': target})

    async def handle_set_confidence(self, request):
        data = await request.json()
        confidence = float(data.get('confidence', 0.25))
        msg = Float32()
        msg.data = confidence
        self.confidence_pub.publish(msg)
        self.settings['conf_threshold'] = confidence
        self.system_state['settings']['conf_threshold'] = confidence
        return web.json_response({'success': True, 'confidence': confidence})

    async def handle_motor_control(self, request):
        data = await request.json()
        if data.get('stop') or data.get('emergency'):
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            self.system_state['motor']['current_command'] = {'linear': 0.0, 'angular': 0.0}
            self._add_alert('CRITICAL', 'Emergency stop activated!')
            return web.json_response({'success': True, 'action': 'stop'})

        linear = float(data.get('linear', 0))
        angular = float(data.get('angular', 0))

        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_vel_pub.publish(twist)
        self.system_state['motor']['current_command'] = {'linear': linear, 'angular': angular}
        return web.json_response({'success': True, 'linear': linear, 'angular': angular})

    async def handle_set_speed(self, request):
        data = await request.json()
        speed = int(data.get('speed', 245))
        self.settings['max_speed'] = speed
        self.system_state['settings']['max_speed'] = speed
        self.system_state['motor']['max_speed'] = speed
        return web.json_response({'success': True, 'speed': speed})

    async def handle_obstacle_config(self, request):
        data = await request.json()
        if 'enabled' in data:
            self.settings['obstacle_avoidance_enabled'] = data['enabled']
            self.system_state['obstacle_avoidance']['enabled'] = data['enabled']
            msg = Bool()
            msg.data = data['enabled']
            self.obstacle_enable_pub.publish(msg)
        if 'min_distance' in data:
            self.settings['min_safe_distance'] = data['min_distance']
            self.system_state['obstacle_avoidance']['min_safe_distance'] = data['min_distance']
            config_msg = String()
            config_msg.data = json.dumps({'min_safe_distance': data['min_distance']})
            self.config_pub.publish(config_msg)
        if 'slow_distance' in data:
            self.settings['slow_down_distance'] = data['slow_distance']
            self.system_state['obstacle_avoidance']['slow_down_distance'] = data['slow_distance']
            config_msg = String()
            config_msg.data = json.dumps({'slow_down_distance': data['slow_distance']})
            self.config_pub.publish(config_msg)
        return web.json_response({'success': True})

    async def handle_websocket(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.websocket_clients.add(ws)

        self._add_log('INFO', 'WebSocket client connected')
        await ws.send_str(json.dumps({
            'type': 'welcome',
            'message': 'Connected to ZRobot',
            'settings': self.settings
        }))

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    await self._handle_ws_message(ws, msg.data)
                elif msg.type == web.WSMsgType.ERROR:
                    self._add_log('ERROR', f'WebSocket error: {ws.exception()}')
        finally:
            self.websocket_clients.remove(ws)
            self._add_log('INFO', 'WebSocket client disconnected')
        return ws

    async def _handle_ws_message(self, ws, data):
        try:
            msg = json.loads(data)
            msg_type = msg.get('type', 'unknown')
            action = msg.get('action')
            payload = msg.get('payload', {})

            if msg_type == 'command':
                if action == 'set_target':
                    target = payload.get('target', 'person')
                    self.settings['target_object'] = target
                    pub_msg = String()
                    pub_msg.data = target
                    self.target_pub.publish(pub_msg)
                    await ws.send_str(json.dumps({'type': 'ack', 'action': 'set_target', 'success': True}))

                elif action == 'set_confidence':
                    conf = float(payload.get('value', 0.45))
                    self.settings['conf_threshold'] = conf
                    pub_msg = Float32()
                    pub_msg.data = conf
                    self.confidence_pub.publish(pub_msg)
                    await ws.send_str(json.dumps({'type': 'ack', 'action': 'set_confidence', 'success': True}))

                elif action == 'motor_control':
                    linear = float(payload.get('linear', 0))
                    angular = float(payload.get('angular', 0))
                    twist = Twist()
                    twist.linear.x = linear
                    twist.angular.z = angular
                    self.cmd_vel_pub.publish(twist)
                    await ws.send_str(json.dumps({'type': 'ack', 'action': 'motor_control', 'success': True}))

                elif action == 'motor_stop':
                    twist = Twist()
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.cmd_vel_pub.publish(twist)
                    await ws.send_str(json.dumps({'type': 'ack', 'action': 'motor_stop', 'success': True}))

                elif action == 'request_full_state':
                    self.update_system_stats()
                    await ws.send_str(json.dumps({
                        'type': 'system_state',
                        'timestamp': int(time.time() * 1000),
                        'data': self.system_state
                    }))

                elif action == 'clear_alerts':
                    self.alerts.clear()
                    await ws.send_str(json.dumps({'type': 'ack', 'action': 'clear_alerts', 'success': True}))

        except json.JSONDecodeError:
            await ws.send_str(json.dumps({'type': 'error', 'message': 'Invalid JSON'}))
        except Exception as e:
            await ws.send_str(json.dumps({'type': 'error', 'message': str(e)}))

    def _get_html(self):
        return """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ZRobot - Центр Управления</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Courier New', Courier, monospace;
            background: #000000;
            color: #00ff41;
            min-height: 100vh;
            overflow-x: hidden;
        }
        /* Scanline effect */
        body::before {
            content: '';
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: repeating-linear-gradient(
                0deg,
                rgba(0, 255, 65, 0.03),
                rgba(0, 255, 65, 0.03) 1px,
                transparent 1px,
                transparent 3px
            );
            pointer-events: none;
            z-index: 1000;
        }
        .header {
            background: #0a0a0a;
            padding: 12px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #00ff41;
            box-shadow: 0 2px 10px rgba(0, 255, 65, 0.3);
        }
        .header h1 {
            font-size: 20px;
            color: #00ff41;
            text-shadow: 0 0 10px #00ff41, 0 0 20px #00ff41;
            letter-spacing: 3px;
        }
        .header-status { display: flex; gap: 15px; align-items: center; font-size: 13px; }
        .status-indicator { display: flex; align-items: center; gap: 8px; padding: 4px 10px; border: 1px solid #00ff41; }
        .status-dot {
            width: 8px; height: 8px;
            animation: pulse 1.5s infinite;
        }
        .status-dot.connected { background: #00ff41; box-shadow: 0 0 8px #00ff41; }
        .status-dot.disconnected { background: #ff0040; box-shadow: 0 0 8px #ff0040; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

        .main-container {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 15px;
            padding: 15px;
            max-width: 1800px;
            margin: 0 auto;
        }
        .video-section { display: flex; flex-direction: column; gap: 12px; }
        .video-container {
            border: 2px solid #00ff41;
            box-shadow: 0 0 15px rgba(0, 255, 65, 0.2);
            background: #0a0a0a;
        }
        #videoCanvas { width: 100%; display: block; background: #000; }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
            gap: 8px;
        }
        .stat-card {
            background: #0a0a0a;
            border: 1px solid #00ff41;
            padding: 10px;
            text-align: center;
        }
        .stat-label {
            font-size: 9px;
            color: #00aa30;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 4px;
        }
        .stat-value {
            font-size: 16px;
            font-weight: bold;
            color: #00ff41;
            text-shadow: 0 0 5px #00ff41;
        }
        .stat-value.success { color: #00ff41; text-shadow: 0 0 5px #00ff41; }
        .stat-value.warning { color: #ffaa00; text-shadow: 0 0 5px #ffaa00; }
        .stat-value.danger { color: #ff0040; text-shadow: 0 0 5px #ff0040; }

        .sidebar { display: flex; flex-direction: column; gap: 12px; }
        .panel {
            background: #0a0a0a;
            border: 1px solid #00ff41;
            padding: 12px;
        }
        .panel-title {
            font-size: 13px;
            font-weight: bold;
            color: #00ff41;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 2px;
            border-bottom: 1px solid #00aa30;
            padding-bottom: 5px;
        }

        .setting-row {
            display: flex; justify-content: space-between; align-items: center;
            padding: 6px 0; border-bottom: 1px solid #003300;
        }
        .setting-row label { font-size: 12px; color: #00aa30; }

        /* Rectangular toggle switch */
        .toggle-switch {
            position: relative; width: 48px; height: 22px;
        }
        .toggle-switch input { opacity: 0; width: 0; height: 0; }
        .toggle-slider {
            position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
            background: #111; border: 1px solid #00ff41; transition: 0.2s;
        }
        .toggle-slider:before {
            position: absolute; content: ""; height: 14px; width: 14px;
            left: 3px; bottom: 3px; background: #00aa30;
            transition: 0.2s;
        }
        .toggle-switch input:checked + .toggle-slider { background: #003300; }
        .toggle-switch input:checked + .toggle-slider:before { transform: translateX(26px); background: #00ff41; box-shadow: 0 0 5px #00ff41; }

        select, input[type="range"] {
            width: 100%; padding: 6px; background: #111;
            border: 1px solid #00ff41;
            color: #00ff41; font-size: 12px; font-family: 'Courier New', monospace;
        }
        select { appearance: none; cursor: pointer; }
        select:focus, input:focus { outline: none; box-shadow: 0 0 5px #00ff41; }
        input[type="range"] { -webkit-appearance: none; height: 8px; cursor: pointer; }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none; width: 14px; height: 14px;
            background: #00ff41; border: 1px solid #00aa30; cursor: pointer;
        }

        /* Rectangular motor buttons */
        .motor-grid {
            display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px;
        }
        .motor-btn {
            padding: 12px; font-size: 16px; font-family: 'Courier New', monospace;
            background: #111; border: 2px solid #00ff41;
            color: #00ff41; cursor: pointer; transition: all 0.15s;
            text-shadow: 0 0 5px #00ff41;
        }
        .motor-btn:hover { background: #00ff41; color: #000; }
        .motor-btn:active { transform: scale(0.95); }
        .motor-btn.stop { border-color: #ff0040; color: #ff0040; }
        .motor-btn.stop:hover { background: #ff0040; color: #000; }

        .obstacle-status { text-align: center; padding: 12px; }
        .obstacle-status.clear { border-color: #00ff41; background: rgba(0, 255, 65, 0.05); }
        .obstacle-status.warning { border-color: #ffaa00; background: rgba(255, 170, 0, 0.05); }
        .obstacle-status.blocked { border-color: #ff0040; background: rgba(255, 0, 64, 0.05); }
        .obstacle-status .panel-title { border-bottom-color: inherit; }

        #lidarCanvas {
            border: 2px solid #00ff41;
            image-rendering: pixelated;
        }

        .logs-panel { max-height: 180px; overflow-y: auto; }
        .log-item {
            padding: 4px 8px; font-size: 10px;
            border-bottom: 1px solid #003300;
        }
        .log-item.INFO { color: #00aa30; }
        .log-item.WARN { color: #ffaa00; }
        .log-item.ERROR { color: #ff0040; }

        .objects-list { max-height: 250px; overflow-y: auto; }
        .object-item {
            display: flex; justify-content: space-between; align-items: center;
            padding: 6px 8px; margin: 3px 0; background: #111;
            border: 1px solid #003300; cursor: pointer; transition: all 0.15s;
        }
        .object-item:hover { background: #003300; border-color: #00ff41; }
        .object-item.active { background: #003300; border-color: #00ff41; box-shadow: 0 0 5px rgba(0, 255, 65, 0.3); }
        .object-name { font-size: 11px; color: #00ff41; }
        .object-conf { font-size: 10px; color: #00aa30; }
        .object-item.active .object-name { color: #00ff41; text-shadow: 0 0 5px #00ff41; }

        .video-container { position: relative; }
        #videoOverlay {
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            cursor: crosshair;
        }

        .sidebar-right {
            display: flex; flex-direction: column; gap: 12px;
            grid-column: 2; grid-row: 1 / 3;
        }
        @media (max-width: 1200px) {
            .main-container { grid-template-columns: 1fr; }
            .sidebar-right { grid-column: 1; grid-row: auto; }
        }
    </style>
</head>
<body>
    <header class="header">
        <h1>> ZROBOT_CTRL_CENTER_</h1>
        <div class="header-status">
            <div class="status-indicator">
                <span class="status-dot" id="wsStatus"></span>
                <span id="wsStatusText">ПОДКЛЮЧЕНИЕ...</span>
            </div>
            <div class="status-indicator">
                <span>АПТАЙМ:</span>
                <span id="uptimeDisplay">0:00:00</span>
            </div>
        </div>
    </header>

    <div class="main-container">
        <div class="video-section">
            <div class="panel">
                <div class="video-container">
                    <canvas id="videoCanvas" width="800" height="480"></canvas>
                    <canvas id="videoOverlay" width="800" height="480"></canvas>
                </div>
            </div>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Цель</div>
                    <div class="stat-value" id="targetDisplay">-</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Найдена</div>
                    <div class="stat-value" id="foundDisplay">-</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Зона</div>
                    <div class="stat-value" id="zoneDisplay">-</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">FPS</div>
                    <div class="stat-value" id="fpsDisplay">0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Задержка</div>
                    <div class="stat-value" id="latencyDisplay">0ms</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Объекты</div>
                    <div class="stat-value" id="detectionsDisplay">0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Трекинг</div>
                    <div class="stat-value" id="trackedDisplay">0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">CPU</div>
                    <div class="stat-value" id="cpuDisplay">0%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">RAM</div>
                    <div class="stat-value" id="memDisplay">0MB</div>
                </div>
            </div>
        </div>

        <div class="sidebar-right">
            <div class="panel obstacle-status" id="obstaclePanel">
                <div class="panel-title">Обход Препятствий</div>
                <div style="font-size: 32px; margin: 8px 0; text-shadow: 0 0 15px currentColor;" id="obstacleIcon">[OK]</div>
                <div id="obstacleText" style="font-size: 14px; letter-spacing: 2px;">ПРОХОД ЯСЕН</div>
                <div style="font-size: 11px; color: #00aa30; margin-top: 6px;">
                    ДИСТАНЦИЯ: <span id="obstacleDistance" style="color: #00ff41;">--</span> м
                </div>
                <div class="setting-row" style="margin-top: 10px;">
                    <label>АКТИВНО</label>
                    <label class="toggle-switch">
                        <input type="checkbox" id="obstacleToggle" checked onchange="toggleObstacle(this.checked)">
                        <span class="toggle-slider"></span>
                    </label>
                </div>
            </div>

            <div class="panel">
                <div class="panel-title">Управление Двигателем</div>
                <div class="stats-grid" style="margin-bottom: 8px;">
                    <div class="stat-card">
                        <div class="stat-label">Статус</div>
                        <div class="stat-value" id="motorStatus">ВЫКЛ</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Скорость</div>
                        <div class="stat-value" id="motorSpeed">0</div>
                    </div>
                </div>
                <div class="motor-grid">
                    <button class="motor-btn" onmousedown="startMotor(0.5, 0)" onmouseup="stopMotor()" ontouchstart="startMotor(0.5, 0)" ontouchend="stopMotor()">▲</button>
                    <button class="motor-btn stop" onclick="sendStop()">■</button>
                    <button class="motor-btn" onmousedown="startMotor(-0.5, 0)" onmouseup="stopMotor()" ontouchstart="startMotor(-0.5, 0)" ontouchend="stopMotor()">▼</button>
                    <button class="motor-btn" onmousedown="startMotor(0, 0.5)" onmouseup="stopMotor()" ontouchstart="startMotor(0, 0.5)" ontouchend="stopMotor()">◄</button>
                    <button class="motor-btn stop" onclick="emergencyStop()">⚠</button>
                    <button class="motor-btn" onmousedown="startMotor(0, -0.5)" onmouseup="stopMotor()" ontouchstart="startMotor(0, -0.5)" ontouchend="stopMotor()">►</button>
                </div>
            </div>

            <div class="panel">
                <div class="panel-title">Лидар - Вид Сверху</div>
                <canvas id="lidarCanvas" width="380" height="380" style="width: 100%; background: #050505;"></canvas>
                <div class="stats-grid" style="margin-top: 8px;">
                    <div class="stat-card">
                        <div class="stat-label">Связь</div>
                        <div class="stat-value" id="lidarConnected">--</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Точки</div>
                        <div class="stat-value" id="lidarPoints">0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Фронт</div>
                        <div class="stat-value" id="lidarFrontDist">--</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Лево</div>
                        <div class="stat-value" id="lidarLeftDist">--</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Право</div>
                        <div class="stat-value" id="lidarRightDist">--</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Тыл</div>
                        <div class="stat-value" id="lidarRearDist">--</div>
                    </div>
                </div>
            </div>

            <div class="panel">
                <div class="panel-title">Настройки</div>
                <div class="setting-row">
                    <label>Цель</label>
                    <select id="targetSelect" style="width: 130px;" onchange="updateSetting('target_object', this.value)">
                        <option value="person">Человек</option>
                        <option value="bicycle">Велосипед</option>
                        <option value="car">Машина</option>
                        <option value="motorcycle">Мотоцикл</option>
                        <option value="airplane">Самолет</option>
                        <option value="bus">Автобус</option>
                        <option value="train">Поезд</option>
                        <option value="truck">Грузовик</option>
                        <option value="boat">Лодка</option>
                        <option value="traffic light">Светофор</option>
                        <option value="fire hydrant">Гидрант</option>
                        <option value="stop sign">Дорожный знак</option>
                        <option value="parking meter">Парковочный счетчик</option>
                        <option value="bench">Скамейка</option>
                        <option value="bird">Птица</option>
                        <option value="cat">Кошка</option>
                        <option value="dog">Собака</option>
                        <option value="horse">Лошадь</option>
                        <option value="sheep">Овца</option>
                        <option value="cow">Корова</option>
                        <option value="elephant">Слон</option>
                        <option value="bear">Медведь</option>
                        <option value="zebra">Зебра</option>
                        <option value="giraffe">Жираф</option>
                        <option value="backpack">Рюкзак</option>
                        <option value="umbrella">Зонт</option>
                        <option value="handbag">Сумка</option>
                        <option value="tie">Галстук</option>
                        <option value="suitcase">Чемодан</option>
                        <option value="frisbee">Фрисби</option>
                        <option value="skis">Лыжи</option>
                        <option value="snowboard">Сноуборд</option>
                        <option value="sports ball">Мяч</option>
                        <option value="kite">Воздушный змей</option>
                        <option value="baseball bat">Бейсбольная бита</option>
                        <option value="baseball glove">Бейсбольная перчатка</option>
                        <option value="skateboard">Скейтборд</option>
                        <option value="surfboard">Серф</option>
                        <option value="tennis racket">Теннисная ракетка</option>
                        <option value="bottle">Бутылка</option>
                        <option value="wine glass">Бокал</option>
                        <option value="cup">Кружка</option>
                        <option value="fork">Вилка</option>
                        <option value="knife">Нож</option>
                        <option value="spoon">Ложка</option>
                        <option value="bowl">Миска</option>
                        <option value="banana">Банан</option>
                        <option value="apple">Яблоко</option>
                        <option value="sandwich">Сэндвич</option>
                        <option value="orange">Апельсин</option>
                        <option value="broccoli">Брокколи</option>
                        <option value="carrot">Морковь</option>
                        <option value="hot dog">Хот-дог</option>
                        <option value="pizza">Пицца</option>
                        <option value="donut">Пончик</option>
                        <option value="cake">Торт</option>
                        <option value="chair">Стул</option>
                        <option value="couch">Диван</option>
                        <option value="potted plant">Растение</option>
                        <option value="bed">Кровать</option>
                        <option value="dining table">Стол</option>
                        <option value="toilet">Туалет</option>
                        <option value="tv">Телевизор</option>
                        <option value="laptop">Ноутбук</option>
                        <option value="mouse">Мышь</option>
                        <option value="remote">Пульт</option>
                        <option value="keyboard">Клавиатура</option>
                        <option value="cell phone">Телефон</option>
                        <option value="microwave">Микроволновка</option>
                        <option value="oven">Духовка</option>
                        <option value="toaster">Тостер</option>
                        <option value="sink">Раковина</option>
                        <option value="refrigerator">Холодильник</option>
                        <option value="book">Книга</option>
                        <option value="clock">Часы</option>
                        <option value="vase">Ваза</option>
                        <option value="scissors">Ножницы</option>
                        <option value="teddy bear">Плюшевый мишка</option>
                        <option value="hair drier">Фен</option>
                        <option value="toothbrush">Зубная щетка</option>
                    </select>
                </div>
                <div class="setting-row">
                    <label>Порог</label>
                    <input type="range" id="confSlider" min="0.1" max="0.9" step="0.05" value="0.45" style="width: 80px;" onchange="updateConfidence(this.value)">
                </div>
                <div class="setting-row">
                    <label>Трекинг</label>
                    <label class="toggle-switch">
                        <input type="checkbox" id="trackingToggle" checked onchange="toggleSetting('enable_tracking', this.checked)">
                        <span class="toggle-slider"></span>
                    </label>
                </div>
                <div class="setting-row">
                    <label>Безоп. Дист.</label>
                    <input type="range" id="safeDistSlider" min="0.1" max="1.0" step="0.1" value="0.3" style="width: 80px;" onchange="updateObstacleParam('min_distance', this.value)">
                </div>
            </div>

            <div class="panel logs-panel" id="logsPanel">
                <div class="panel-title">Логи</div>
                <div id="logsList"></div>
            </div>

            <div class="panel">
                <div class="panel-title">Обнаруженные Объекты</div>
                <div class="objects-list" id="objectsList">
                    <div style="font-size: 11px; color: #00aa30; text-align: center; padding: 10px;">Нет данных</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let reconnectAttempts = 0;

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(protocol + '//' + window.location.host + '/ws');

            ws.onopen = function() {
                document.getElementById('wsStatus').className = 'status-dot connected';
                document.getElementById('wsStatusText').textContent = 'ПОДКЛЮЧЕНО';
                reconnectAttempts = 0;
            };

            ws.onclose = function() {
                document.getElementById('wsStatus').className = 'status-dot disconnected';
                document.getElementById('wsStatusText').textContent = 'ОТКЛЮЧЕНО';
                reconnectAttempts++;
                setTimeout(connectWebSocket, Math.min(reconnectAttempts * 1000, 5000));
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'system_state') {
                    updateUI(data.data);
                } else if (data.type === 'welcome') {
                    updateSettingsUI(data.settings);
                }
            };

            ws.onerror = function() { ws.close(); };
        }

        function updateUI(state) {
            window._lastState = state;
            const det = state.detection || {};
            document.getElementById('targetDisplay').textContent = det.target || '-';

            const foundEl = document.getElementById('foundDisplay');
            foundEl.textContent = det.found ? 'ДА' : 'НЕТ';
            foundEl.className = 'stat-value ' + (det.found ? 'success' : 'danger');

            document.getElementById('zoneDisplay').textContent = det.zone || '-';
            document.getElementById('fpsDisplay').textContent = (det.fps || 0).toFixed(1);
            document.getElementById('latencyDisplay').textContent = (det.inference_time_ms || 0).toFixed(1) + 'ms';
            document.getElementById('detectionsDisplay').textContent = det.detections_count || 0;
            document.getElementById('trackedDisplay').textContent = det.tracked_count || 0;

            // Update detected objects list
            updateObjectsList(det.objects || [], det.target || '-');

            // Draw bounding boxes on overlay
            drawDetectionBoxes(det.objects || []);

            const sys = state.system || {};
            document.getElementById('uptimeDisplay').textContent = formatUptime(sys.uptime_seconds || 0);
            document.getElementById('cpuDisplay').textContent = (sys.cpu_usage || 0).toFixed(0) + '%';
            document.getElementById('memDisplay').textContent = (sys.memory_usage_mb || 0) + 'MB';

            const motor = state.motor || {};
            const motorStatusEl = document.getElementById('motorStatus');
            motorStatusEl.textContent = motor.enabled ? 'ВКЛ' : 'ВЫКЛ';
            motorStatusEl.className = 'stat-value ' + (motor.enabled ? 'success' : 'danger');
            document.getElementById('motorSpeed').textContent = motor.speed || 0;

            const obstacle = state.obstacle_avoidance || {};
            const obsPanel = document.getElementById('obstaclePanel');
            const obsIcon = document.getElementById('obstacleIcon');
            const obsText = document.getElementById('obstacleText');

            obsPanel.className = 'panel obstacle-status';
            if (obstacle.obstacle_detected) {
                obsPanel.classList.add('blocked');
                obsIcon.textContent = '[!!]';
                obsIcon.style.color = '#ff0040';
                obsText.textContent = 'БЛОКИРОВКА';
                obsText.style.color = '#ff0040';
            } else if (obstacle.min_distance_m < obstacle.slow_down_distance) {
                obsPanel.classList.add('warning');
                obsIcon.textContent = '[!]';
                obsIcon.style.color = '#ffaa00';
                obsText.textContent = 'ЗАМЕДЛЕНИЕ';
                obsText.style.color = '#ffaa00';
            } else {
                obsPanel.classList.add('clear');
                obsIcon.textContent = '[OK]';
                obsIcon.style.color = '#00ff41';
                obsText.textContent = 'ПРОХОД ЯСЕН';
                obsText.style.color = '#00ff41';
            }

            document.getElementById('obstacleDistance').textContent =
                obstacle.min_distance_m < 10 ? obstacle.min_distance_m.toFixed(2) : '--';
            document.getElementById('obstacleToggle').checked = obstacle.enabled;

            const lidar = state.lidar || {};
            const lidarConn = document.getElementById('lidarConnected');
            lidarConn.textContent = lidar.connected ? 'ЕСТЬ' : 'НЕТ';
            lidarConn.className = 'stat-value ' + (lidar.connected ? 'success' : 'danger');
            document.getElementById('lidarPoints').textContent = lidar.points || 0;

            // Render lidar map
            const lidarCanvas = document.getElementById('lidarCanvas');
            if (lidarCanvas && lidar.scan_data) {
                renderLidarMap(lidarCanvas, lidar.scan_data);
            }

            if (state.logs && state.logs.length > 0) {
                const logsList = document.getElementById('logsList');
                logsList.innerHTML = state.logs.slice(-10).map(log =>
                    `<div class="log-item ${log.level}">[${new Date(log.timestamp).toLocaleTimeString()}] ${log.message}</div>`
                ).join('');
            }
        }

        function updateSettingsUI(settings) {
            document.getElementById('targetSelect').value = settings.target_object || 'person';
            document.getElementById('confSlider').value = settings.conf_threshold || 0.45;
            document.getElementById('trackingToggle').checked = settings.enable_tracking !== false;
            document.getElementById('obstacleToggle').checked = settings.obstacle_avoidance_enabled !== false;
            document.getElementById('safeDistSlider').value = settings.min_safe_distance || 0.3;
        }

        function renderLidarMap(canvas, scanData) {
            const ctx = canvas.getContext('2d');
            const width = canvas.width;
            const height = canvas.height;
            const cx = width / 2;
            const cy = height / 2;
            const maxRange = scanData.range_max || 5.0;
            const angleOffsetDeg = scanData.angle_offset_deg || 0;
            // Show only up to 3m for better detail, stretch scale
            const displayMaxRange = Math.min(maxRange, 3.0);
            const padding = 35;
            const scale = (Math.min(width, height) / 2 - padding) / displayMaxRange;

            const obstacle = (window._lastState && window._lastState.obstacle_avoidance) || {};
            const isBlocked = obstacle.obstacle_detected || false;
            const minDist = obstacle.min_distance_m || 999.0;
            const slowDist = obstacle.slow_down_distance || 0.5;

            // Border color based on obstacle status
            if (isBlocked) {
                canvas.style.borderColor = '#ff0040';
                canvas.style.boxShadow = '0 0 15px rgba(255, 0, 64, 0.4)';
            } else if (minDist < slowDist) {
                canvas.style.borderColor = '#ffaa00';
                canvas.style.boxShadow = '0 0 15px rgba(255, 170, 0, 0.4)';
            } else {
                canvas.style.borderColor = '#00ff41';
                canvas.style.boxShadow = '0 0 10px rgba(0, 255, 65, 0.2)';
            }

            // Background
            ctx.fillStyle = '#050505';
            ctx.fillRect(0, 0, width, height);

            // Grid circles with distance labels
            const gridSteps = 5;
            ctx.lineWidth = 1;
            ctx.font = '9px Courier New';
            ctx.textAlign = 'left';
            for (let i = 1; i <= gridSteps; i++) {
                const radius = (i * displayMaxRange / gridSteps) * scale;
                ctx.strokeStyle = `rgba(0, 255, 65, ${0.1 + i * 0.03})`;
                ctx.beginPath();
                ctx.arc(cx, cy, radius, 0, Math.PI * 2);
                ctx.stroke();
                ctx.fillStyle = 'rgba(0, 255, 65, 0.5)';
                ctx.fillText((i * displayMaxRange / gridSteps).toFixed(1) + 'м', cx + radius + 2, cy + 3);
            }

            // Cross lines (axes)
            ctx.strokeStyle = 'rgba(0, 255, 65, 0.2)';
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(cx, 0); ctx.lineTo(cx, height);
            ctx.moveTo(0, cy); ctx.lineTo(width, cy);
            ctx.stroke();
            ctx.setLineDash([]);

            // Zone sector lines (with offset applied)
            // Display: top=front, right=right, left=left, bottom=rear
            const sectors = [
                { angle: 90, label: 'ФРОНТ', color: 'rgba(0, 255, 65, 0.3)' },
                { angle: -90, label: 'ТЫЛ', color: 'rgba(255, 170, 0, 0.25)' },
                { angle: 180, label: 'ЛЕВО', color: 'rgba(0, 200, 50, 0.25)' },
                { angle: 0, label: 'ПРАВО', color: 'rgba(0, 200, 50, 0.25)' },
            ];

            sectors.forEach(s => {
                const rad = s.angle * Math.PI / 180;
                ctx.strokeStyle = s.color;
                ctx.lineWidth = 1;
                ctx.setLineDash([3, 3]);
                ctx.beginPath();
                ctx.moveTo(cx, cy);
                ctx.lineTo(cx + Math.cos(rad) * Math.max(width, height), cy + Math.sin(rad) * Math.max(width, height));
                ctx.stroke();
            });
            ctx.setLineDash([]);

            // Direction labels (rectangular boxes)
            const drawLabel = (text, x, y, color) => {
                ctx.font = 'bold 10px Courier New';
                const tw = ctx.measureText(text).width;
                ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                ctx.fillRect(x - tw/2 - 3, y - 7, tw + 6, 14);
                ctx.strokeStyle = color;
                ctx.lineWidth = 1;
                ctx.strokeRect(x - tw/2 - 3, y - 7, tw + 6, 14);
                ctx.fillStyle = color;
                ctx.textAlign = 'center';
                ctx.fillText(text, x, y + 3);
            };

            drawLabel('▲ ФРОНТ', cx, height - 8, '#00ff41');
            drawLabel('▼ ТЫЛ', cx, 15, '#ffaa00');
            drawLabel('◄ ЛЕВО', 55, cy + 3, '#00c832');
            drawLabel('ПРАВО ►', width - 55, cy + 3, '#00c832');

            // Parse scan data
            const ranges = scanData.ranges || [];
            const angleMin = scanData.angle_min || 0;
            const angleInc = scanData.angle_increment || 0.017;

            // Build points - apply angle offset to rotate map so front is at top
            const points = [];
            for (let i = 0; i < ranges.length; i++) {
                const r = ranges[i];
                if (r === null || r === undefined || r <= 0) continue;
                const rawAngle = angleMin + i * angleInc;
                // Apply offset: shift angle so display front aligns with top of canvas
                const adjustedAngle = rawAngle + (angleOffsetDeg * Math.PI / 180);
                const dist = Math.min(r, displayMaxRange);
                // Skip points beyond display range
                if (r > displayMaxRange) continue;
                const x = cx + Math.cos(adjustedAngle) * dist * scale;
                const y = cy + Math.sin(adjustedAngle) * dist * scale;
                points.push({x, y, dist, raw: r});
            }

            if (points.length === 0) {
                ctx.fillStyle = '#ff0040';
                ctx.font = '12px Courier New';
                ctx.textAlign = 'center';
                ctx.fillText('НЕТ ДАННЫХ ЛИДАРА', cx, cy);
                return;
            }

            // Draw filled polygon for obstacle shape
            if (points.length > 2) {
                // Outer glow
                ctx.strokeStyle = 'rgba(0, 255, 65, 0.2)';
                ctx.lineWidth = 5;
                ctx.beginPath();
                ctx.moveTo(points[0].x, points[0].y);
                for (let i = 1; i < points.length; i++) {
                    ctx.lineTo(points[i].x, points[i].y);
                }
                ctx.closePath();
                ctx.stroke();

                // Gradient fill
                const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, displayMaxRange * scale);
                gradient.addColorStop(0, 'rgba(0, 255, 65, 0.35)');
                gradient.addColorStop(0.6, 'rgba(0, 200, 50, 0.2)');
                gradient.addColorStop(1, 'rgba(0, 100, 25, 0.1)');
                ctx.fillStyle = gradient;
                ctx.beginPath();
                ctx.moveTo(points[0].x, points[0].y);
                for (let i = 1; i < points.length; i++) {
                    ctx.lineTo(points[i].x, points[i].y);
                }
                ctx.closePath();
                ctx.fill();

                // Main outline
                ctx.strokeStyle = '#00ff41';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(points[0].x, points[0].y);
                for (let i = 1; i < points.length; i++) {
                    ctx.lineTo(points[i].x, points[i].y);
                }
                ctx.closePath();
                ctx.stroke();
            }

            // Draw individual points colored by distance (sparser for performance)
            const step = Math.max(1, Math.floor(points.length / 300)); // Max 300 dots
            for (let idx = 0; idx < points.length; idx += step) {
                const p = points[idx];
                const intensity = 1 - (p.dist / displayMaxRange);
                const green = Math.floor(180 + intensity * 75);
                const alpha = 0.4 + intensity * 0.6;
                const size = 1.5 + intensity * 2;
                ctx.fillStyle = `rgba(0, ${green}, 40, ${alpha})`;
                ctx.fillRect(p.x - size/2, p.y - size/2, size, size); // Square dots for hacker style
            }

            // Zone distance arcs
            const frontDist = obstacle.front_distance || obstacle.min_distance_m || 999.0;
            const leftDist = obstacle.left_distance || 999.0;
            const rightDist = obstacle.right_distance || 999.0;
            const rearDist = obstacle.rear_distance || 999.0;

            // Update stat cards
            const setDist = (id, dist) => {
                const el = document.getElementById(id);
                if (el) el.textContent = dist < 10 ? dist.toFixed(2) + 'м' : '--';
            };
            setDist('lidarFrontDist', frontDist);
            setDist('lidarLeftDist', leftDist);
            setDist('lidarRightDist', rightDist);
            setDist('lidarRearDist', rearDist);

            // Draw distance arcs on canvas
            const drawArc = (angle, dist, color) => {
                if (dist >= 10) return;
                const rad = angle * Math.PI / 180;
                const r = dist * scale;
                ctx.strokeStyle = color;
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.arc(cx, cy, r, rad - 0.25, rad + 0.25);
                ctx.stroke();
            };

            drawArc(90, frontDist, '#00ff41');
            drawArc(-90, rearDist, '#ffaa00');
            drawArc(180, leftDist, '#00c832');
            drawArc(0, rightDist, '#00c832');

            // Center robot indicator - square for hacker style
            ctx.fillStyle = '#ff0040';
            ctx.fillRect(cx - 6, cy - 6, 12, 12);
            ctx.fillStyle = '#000';
            ctx.fillRect(cx - 3, cy - 3, 6, 6);
            ctx.fillStyle = '#00ff41';
            ctx.fillRect(cx - 1, cy - 1, 2, 2);

            // Front direction arrow
            ctx.fillStyle = '#00ff41';
            ctx.beginPath();
            ctx.moveTo(cx, cy - 18);
            ctx.lineTo(cx - 5, cy - 11);
            ctx.lineTo(cx + 5, cy - 11);
            ctx.closePath();
            ctx.fill();
        }

        function updateObjectsList(objects, currentTarget) {
            const listEl = document.getElementById('objectsList');
            if (!objects || objects.length === 0) {
                listEl.innerHTML = '<div style="font-size: 11px; color: #00aa30; text-align: center; padding: 10px;">Нет объектов</div>';
                return;
            }
            listEl.innerHTML = objects.map((obj, idx) => {
                const isActive = obj.class === currentTarget;
                return `<div class="object-item ${isActive ? 'active' : ''}" onclick="setTargetFromObject('${obj.class}')">
                    <span class="object-name">${obj.class}</span>
                    <span class="object-conf">${(obj.confidence * 100).toFixed(0)}%</span>
                </div>`;
            }).join('');
        }

        function setTargetFromObject(className) {
            document.getElementById('targetSelect').value = className;
            updateSetting('target_object', className);
        }

        function drawDetectionBoxes(objects) {
            const canvas = document.getElementById('videoOverlay');
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (!objects || objects.length === 0) return;

            const currentTarget = (window._lastState && window._lastState.detection && window._lastState.detection.target) || '-';

            objects.forEach(obj => {
                const isTarget = obj.class === currentTarget;
                const color = isTarget ? '#00ff41' : '#00aa30';
                const lineWidth = isTarget ? 3 : 1;

                ctx.strokeStyle = color;
                ctx.lineWidth = lineWidth;
                ctx.strokeRect(obj.x, obj.y, obj.w, obj.h);

                // Label background
                ctx.font = '11px Courier New';
                const labelText = `${obj.class} ${(obj.confidence * 100).toFixed(0)}%`;
                const textWidth = ctx.measureText(labelText).width;
                ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                ctx.fillRect(obj.x, obj.y - 16, textWidth + 8, 16);

                // Label text
                ctx.fillStyle = color;
                ctx.fillText(labelText, obj.x + 4, obj.y - 4);
            });
        }

        // Click handling on video overlay
        document.addEventListener('DOMContentLoaded', function() {
            const overlay = document.getElementById('videoOverlay');
            overlay.addEventListener('click', function(e) {
                const rect = overlay.getBoundingClientRect();
                const scaleX = overlay.width / rect.width;
                const scaleY = overlay.height / rect.height;
                const clickX = (e.clientX - rect.left) * scaleX;
                const clickY = (e.clientY - rect.top) * scaleY;

                const objects = (window._lastState && window._lastState.detection && window._lastState.detection.objects) || [];

                // Find object that was clicked
                for (const obj of objects) {
                    if (clickX >= obj.x && clickX <= obj.x + obj.w &&
                        clickY >= obj.y && clickY <= obj.y + obj.h) {
                        setTargetFromObject(obj.class);
                        return;
                    }
                }
            });
        });

        function formatUptime(seconds) {
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = seconds % 60;
            return `${h}:${m.toString().padStart(2,'0')}:${s.toString().padStart(2,'0')}`;
        }

        function updateSetting(key, value) {
            fetch('/api/settings', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({[key]: value})
            });
        }

        function updateConfidence(value) {
            fetch('/api/confidence', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({confidence: parseFloat(value)})
            });
        }

        function toggleSetting(key, value) {
            fetch('/api/settings', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({[key]: value})
            });
        }

        function toggleObstacle(value) {
            fetch('/api/obstacle', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({enabled: value})
            });
        }

        function updateObstacleParam(param, value) {
            fetch('/api/obstacle', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({[param]: parseFloat(value)})
            });
        }

        let motorInterval = null;
        let lastMotorCmd = {linear: 0, angular: 0};

        function sendMotorOnce(linear, angular) {
            lastMotorCmd = {linear: linear, angular: angular};
            fetch('/api/motor', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(lastMotorCmd)
            }).catch(() => {});
        }

        function startMotor(linear, angular) {
            stopMotor();
            sendMotorOnce(linear, angular);
            motorInterval = setInterval(() => {
                sendMotorOnce(lastMotorCmd.linear, lastMotorCmd.angular);
            }, 100);
        }

        function stopMotor() {
            if (motorInterval) {
                clearInterval(motorInterval);
                motorInterval = null;
            }
            sendMotorOnce(0, 0);
        }

        function sendMotor(linear, angular) { startMotor(linear, angular); }
        function sendStop() { stopMotor(); }

        function emergencyStop() {
            fetch('/api/motor', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({emergency: true})
            });
        }

        function updateVideo() {
            const canvas = document.getElementById('videoCanvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();
            img.onload = function() { ctx.drawImage(img, 0, 0, canvas.width, canvas.height); };
            img.src = '/api/frame?t=' + Date.now();
            setTimeout(updateVideo, 100);
        }

        connectWebSocket();
        updateVideo();
    </script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description='ZRobot Web Interface')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    parser.add_argument('--rate', type=float, default=5.0, help='State update rate (Hz)')
    args, unknown = parser.parse_known_args()

    rclpy.init(args=unknown)
    node = WebInterfaceNode()
    node.host = args.host
    node.port = args.port
    node.state_update_rate = args.rate

    server_thread = threading.Thread(target=node.run_server, daemon=True)
    server_thread.start()

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except rclpy._rclpy_pybind11.RCLError:
            pass  # Already shut down by launch system


if __name__ == '__main__':
    main()
