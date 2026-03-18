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
                'error': None
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
            'max_speed': 245,
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
        if msg.detections:
            scores = [det.results[0].hypothesis.score for det in msg.detections if det.results]
            if scores:
                self.system_state['detection']['confidence'] = max(scores)

    def tracked_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.system_state['detection']['tracked_count'] = data.get('tracked', 0)
        except:
            pass

    def obstacle_callback(self, msg):
        self.system_state['obstacle_avoidance']['obstacle_detected'] = msg.data
        if msg.data:
            self._add_alert('WARNING', 'Obstacle detected!')

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
            self.system_state['motor'].update({
                'enabled': data.get('enabled', True),
                'speed': data.get('speed', 0),
                'status': data.get('status', 'stopped'),
                'error': data.get('error')
            })
            self.motor_history.append({
                'timestamp': time.time(),
                'speed': data.get('speed', 0),
                'enabled': data.get('enabled', True)
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
        alert = {
            'level': level,
            'message': message,
            'timestamp': int(time.time() * 1000)
        }
        self.alerts.append(alert)

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
        if 'slow_distance' in data:
            self.settings['slow_down_distance'] = data['slow_distance']
            self.system_state['obstacle_avoidance']['slow_down_distance'] = data['slow_distance']
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
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ZRobot Control Center</title>
    <style>
        :root {
            --primary: #00d9ff;
            --secondary: #7b2cbf;
            --success: #00ff88;
            --warning: #ffaa00;
            --danger: #ff4466;
            --bg-dark: #0f0f1a;
            --bg-card: #1a1a2e;
            --bg-panel: #16213e;
            --text-primary: #ffffff;
            --text-secondary: #a0a0b0;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, var(--bg-dark) 0%, #1a1a3e 100%);
            color: var(--text-primary);
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(90deg, var(--bg-card) 0%, var(--bg-panel) 100%);
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid var(--primary);
        }
        .header h1 {
            font-size: 22px;
            background: linear-gradient(90deg, var(--primary), var(--success));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header-status { display: flex; gap: 15px; align-items: center; }
        .status-indicator {
            display: flex; align-items: center; gap: 8px;
            padding: 6px 12px; background: var(--bg-dark);
            border-radius: 15px; font-size: 13px;
        }
        .status-dot {
            width: 8px; height: 8px; border-radius: 50%;
            animation: pulse 2s infinite;
        }
        .status-dot.connected { background: var(--success); }
        .status-dot.disconnected { background: var(--danger); }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        
        .main-container {
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 20px;
            padding: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }
        .video-section { display: flex; flex-direction: column; gap: 15px; }
        .video-container {
            background: var(--bg-card); border-radius: 12px; overflow: hidden;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        }
        #videoCanvas { width: 100%; display: block; background: #000; }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
        }
        .stat-card {
            background: var(--bg-panel); border-radius: 8px; padding: 12px;
            text-align: center; border: 1px solid rgba(0,217,255,0.15);
        }
        .stat-label { font-size: 10px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; }
        .stat-value { font-size: 18px; font-weight: bold; color: var(--primary); margin-top: 4px; }
        .stat-value.success { color: var(--success); }
        .stat-value.warning { color: var(--warning); }
        .stat-value.danger { color: var(--danger); }
        
        .sidebar { display: flex; flex-direction: column; gap: 15px; }
        .panel {
            background: var(--bg-card); border-radius: 12px; padding: 15px;
            border: 1px solid rgba(0,217,255,0.15);
        }
        .panel-title { font-size: 14px; font-weight: 600; color: var(--primary); margin-bottom: 12px; }
        
        .setting-row {
            display: flex; justify-content: space-between; align-items: center;
            padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .setting-row label { font-size: 13px; color: var(--text-secondary); }
        
        .toggle-switch {
            position: relative; width: 44px; height: 24px;
        }
        .toggle-switch input { opacity: 0; width: 0; height: 0; }
        .toggle-slider {
            position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
            background: var(--bg-dark); border-radius: 24px; transition: 0.3s;
            border: 1px solid rgba(0,217,255,0.3);
        }
        .toggle-slider:before {
            position: absolute; content: ""; height: 16px; width: 16px;
            left: 3px; bottom: 3px; background: var(--text-secondary);
            border-radius: 50%; transition: 0.3s;
        }
        .toggle-switch input:checked + .toggle-slider { background: var(--primary); }
        .toggle-switch input:checked + .toggle-slider:before { transform: translateX(20px); background: #fff; }
        
        select, input[type="range"] {
            width: 100%; padding: 8px; background: var(--bg-dark);
            border: 1px solid rgba(0,217,255,0.3); border-radius: 6px;
            color: var(--text-primary); font-size: 13px;
        }
        select:focus, input:focus { outline: none; border-color: var(--primary); }
        
        .motor-grid {
            display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;
        }
        .motor-btn {
            padding: 15px; font-size: 18px; background: var(--bg-dark);
            border: 2px solid var(--primary); border-radius: 8px;
            color: var(--primary); cursor: pointer; transition: all 0.2s;
        }
        .motor-btn:hover { background: var(--primary); color: var(--bg-dark); }
        .motor-btn.stop { border-color: var(--danger); color: var(--danger); }
        .motor-btn.stop:hover { background: var(--danger); color: #fff; }
        
        .detections-list { max-height: 150px; overflow-y: auto; }
        .detection-item {
            display: flex; justify-content: space-between; align-items: center;
            padding: 8px; background: var(--bg-dark); border-radius: 5px;
            margin-bottom: 5px; border-left: 3px solid var(--primary);
        }
        .detection-class { font-weight: 600; color: var(--primary); font-size: 13px; }
        .detection-score { background: var(--primary); padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: bold; color: var(--bg-dark); }
        
        .logs-panel { max-height: 200px; overflow-y: auto; }
        .log-item { padding: 5px 8px; font-size: 11px; font-family: monospace; border-bottom: 1px solid rgba(255,255,255,0.05); }
        .log-item.INFO { color: var(--text-secondary); }
        .log-item.WARN { color: var(--warning); }
        .log-item.ERROR { color: var(--danger); }
        
        .alert-banner {
            background: linear-gradient(90deg, var(--danger), #ff0044);
            color: white; padding: 10px 20px; border-radius: 8px;
            margin-bottom: 15px; display: none; animation: slideIn 0.3s;
        }
        .alert-banner.show { display: block; }
        @keyframes slideIn { from { transform: translateY(-20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
        
        .sidebar-right {
            display: flex; flex-direction: column; gap: 15px;
            grid-column: 2; grid-row: 1 / 3;
        }
        .obstacle-status { text-align: center; padding: 15px; }
        .obstacle-status.clear { background: rgba(0,255,136,0.1); border: 1px solid var(--success); border-radius: 8px; }
        .obstacle-status.warning { background: rgba(255,170,0,0.1); border: 1px solid var(--warning); border-radius: 8px; }
        .obstacle-status.blocked { background: rgba(255,68,102,0.1); border: 1px solid var(--danger); border-radius: 8px; }
        
        @media (max-width: 1200px) {
            .main-container { grid-template-columns: 1fr; }
            .sidebar-right { grid-column: 1; grid-row: auto; }
        }
    </style>
</head>
<body>
    <header class="header">
        <h1>ZRobot Control Center</h1>
        <div class="header-status">
            <div class="status-indicator">
                <span class="status-dot" id="wsStatus"></span>
                <span id="wsStatusText">Connecting...</span>
            </div>
            <div class="status-indicator">
                <span>Uptime:</span>
                <span id="uptimeDisplay">0:00:00</span>
            </div>
        </div>
    </header>

    <div class="alert-banner" id="alertBanner"></div>

    <div class="main-container">
        <div class="video-section">
            <div class="video-container">
                <canvas id="videoCanvas" width="800" height="480"></canvas>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Target</div>
                    <div class="stat-value" id="targetDisplay">-</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Found</div>
                    <div class="stat-value" id="foundDisplay">-</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Zone</div>
                    <div class="stat-value" id="zoneDisplay">-</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">FPS</div>
                    <div class="stat-value" id="fpsDisplay">0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Inference</div>
                    <div class="stat-value" id="latencyDisplay">0ms</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Detections</div>
                    <div class="stat-value" id="detectionsDisplay">0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Tracked</div>
                    <div class="stat-value" id="trackedDisplay">0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">CPU</div>
                    <div class="stat-value" id="cpuDisplay">0%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Memory</div>
                    <div class="stat-value" id="memDisplay">0MB</div>
                </div>
            </div>
        </div>

        <div class="sidebar-right">
            <div class="panel obstacle-status" id="obstaclePanel">
                <div class="panel-title">Obstacle Avoidance</div>
                <div style="font-size: 28px; margin: 10px 0;" id="obstacleIcon">✓</div>
                <div id="obstacleText">Clear</div>
                <div style="font-size: 12px; color: var(--text-secondary); margin-top: 8px;">
                    Distance: <span id="obstacleDistance">--</span>m
                </div>
                <div class="setting-row" style="margin-top: 15px;">
                    <label>Enabled</label>
                    <label class="toggle-switch">
                        <input type="checkbox" id="obstacleToggle" checked onchange="toggleObstacle(this.checked)">
                        <span class="toggle-slider"></span>
                    </label>
                </div>
            </div>

            <div class="panel">
                <div class="panel-title">Motor Status</div>
                <div class="stats-grid" style="margin-bottom: 10px;">
                    <div class="stat-card">
                        <div class="stat-label">Status</div>
                        <div class="stat-value" id="motorStatus">OFF</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Speed</div>
                        <div class="stat-value" id="motorSpeed">0</div>
                    </div>
                </div>
                <div class="motor-grid">
                    <button class="motor-btn" onmousedown="sendMotor(0, 0.5)" onmouseup="sendStop()">↑</button>
                    <button class="motor-btn stop" onclick="sendStop()">■</button>
                    <button class="motor-btn" onmousedown="sendMotor(0, -0.5)" onmouseup="sendStop()">↓</button>
                    <button class="motor-btn" onmousedown="sendMotor(-0.3, 0)" onmouseup="sendStop()">←</button>
                    <button class="motor-btn stop" onclick="emergencyStop()">⚠</button>
                    <button class="motor-btn" onmousedown="sendMotor(0.3, 0)" onmouseup="sendStop()">→</button>
                </div>
            </div>

            <div class="panel">
                <div class="panel-title">Lidar</div>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Connected</div>
                        <div class="stat-value" id="lidarConnected">--</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Points</div>
                        <div class="stat-value" id="lidarPoints">0</div>
                    </div>
                </div>
            </div>

            <div class="panel">
                <div class="panel-title">Settings</div>
                <div class="setting-row">
                    <label>Target</label>
                    <select id="targetSelect" style="width: 120px;" onchange="updateSetting('target_object', this.value)">
                        <option value="person">Person</option>
                        <option value="car">Car</option>
                        <option value="dog">Dog</option>
                        <option value="cat">Cat</option>
                        <option value="bicycle">Bicycle</option>
                    </select>
                </div>
                <div class="setting-row">
                    <label>Confidence</label>
                    <input type="range" id="confSlider" min="0.1" max="0.9" step="0.05" value="0.45" style="width: 80px;" onchange="updateConfidence(this.value)">
                </div>
                <div class="setting-row">
                    <label>Tracking</label>
                    <label class="toggle-switch">
                        <input type="checkbox" id="trackingToggle" checked onchange="toggleSetting('enable_tracking', this.checked)">
                        <span class="toggle-slider"></span>
                    </label>
                </div>
                <div class="setting-row">
                    <label>Safe Distance</label>
                    <input type="range" id="safeDistSlider" min="0.1" max="1.0" step="0.1" value="0.3" style="width: 80px;" onchange="updateObstacleParam('min_distance', this.value)">
                </div>
            </div>

            <div class="panel logs-panel" id="logsPanel">
                <div class="panel-title">Logs</div>
                <div id="logsList"></div>
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
                document.getElementById('wsStatusText').textContent = 'Connected';
                reconnectAttempts = 0;
            };

            ws.onclose = function() {
                document.getElementById('wsStatus').className = 'status-dot disconnected';
                document.getElementById('wsStatusText').textContent = 'Disconnected';
                reconnectAttempts++;
                setTimeout(connectWebSocket, Math.min(reconnectAttempts * 1000, 5000));
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'system_state') {
                    updateUI(data.data);
                } else if (data.type === 'welcome') {
                    updateSettingsUI(data.settings);
                } else if (data.type === 'alert') {
                    showAlert(data.message);
                }
            };

            ws.onerror = function() { ws.close(); };
        }

        function updateUI(state) {
            const det = state.detection || {};
            document.getElementById('targetDisplay').textContent = det.target || '-';
            
            const foundEl = document.getElementById('foundDisplay');
            foundEl.textContent = det.found ? 'YES' : 'NO';
            foundEl.className = 'stat-value ' + (det.found ? 'success' : 'danger');
            
            document.getElementById('zoneDisplay').textContent = det.zone || '-';
            document.getElementById('fpsDisplay').textContent = (det.fps || 0).toFixed(1);
            document.getElementById('latencyDisplay').textContent = (det.inference_time_ms || 0).toFixed(1) + 'ms';
            document.getElementById('detectionsDisplay').textContent = det.detections_count || 0;
            document.getElementById('trackedDisplay').textContent = det.tracked_count || 0;

            const sys = state.system || {};
            document.getElementById('uptimeDisplay').textContent = formatUptime(sys.uptime_seconds || 0);
            document.getElementById('cpuDisplay').textContent = (sys.cpu_usage || 0).toFixed(0) + '%';
            document.getElementById('memDisplay').textContent = (sys.memory_usage_mb || 0) + 'MB';

            const motor = state.motor || {};
            const motorStatus = document.getElementById('motorStatus');
            motorStatus.textContent = motor.enabled ? 'ON' : 'OFF';
            motorStatus.className = 'stat-value ' + (motor.enabled ? 'success' : 'danger');
            document.getElementById('motorSpeed').textContent = motor.speed || 0;

            const obstacle = state.obstacle_avoidance || {};
            const obsPanel = document.getElementById('obstaclePanel');
            const obsIcon = document.getElementById('obstacleIcon');
            const obsText = document.getElementById('obstacleText');
            
            obsPanel.className = 'panel obstacle-status';
            if (obstacle.obstacle_detected) {
                obsPanel.classList.add('blocked');
                obsIcon.textContent = '⚠';
                obsText.textContent = 'BLOCKED';
            } else if (obstacle.min_distance_m < obstacle.slow_down_distance) {
                obsPanel.classList.add('warning');
                obsIcon.textContent = '⚡';
                obsText.textContent = 'SLOW';
            } else {
                obsPanel.classList.add('clear');
                obsIcon.textContent = '✓';
                obsText.textContent = 'CLEAR';
            }
            
            document.getElementById('obstacleDistance').textContent = 
                obstacle.min_distance_m < 10 ? obstacle.min_distance_m.toFixed(2) : '--';
            document.getElementById('obstacleToggle').checked = obstacle.enabled;

            const lidar = state.lidar || {};
            const lidarConn = document.getElementById('lidarConnected');
            lidarConn.textContent = lidar.connected ? 'YES' : 'NO';
            lidarConn.className = 'stat-value ' + (lidar.connected ? 'success' : 'danger');
            document.getElementById('lidarPoints').textContent = lidar.points || 0;

            if (state.logs && state.logs.length > 0) {
                const logsList = document.getElementById('logsList');
                logsList.innerHTML = state.logs.slice(-10).map(log => 
                    `<div class="log-item ${log.level}">[${new Date(log.timestamp).toLocaleTimeString()}] ${log.message}</div>`
                ).join('');
            }

            if (state.alerts && state.alerts.length > 0) {
                showAlert(state.alerts[state.alerts.length - 1].message);
            }
        }

        function updateSettingsUI(settings) {
            document.getElementById('targetSelect').value = settings.target_object || 'person';
            document.getElementById('confSlider').value = settings.conf_threshold || 0.45;
            document.getElementById('trackingToggle').checked = settings.enable_tracking !== false;
            document.getElementById('obstacleToggle').checked = settings.obstacle_avoidance_enabled !== false;
            document.getElementById('safeDistSlider').value = settings.min_safe_distance || 0.3;
        }

        function formatUptime(seconds) {
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = seconds % 60;
            return `${h}:${m.toString().padStart(2,'0')}:${s.toString().padStart(2,'0')}`;
        }

        function showAlert(message) {
            const banner = document.getElementById('alertBanner');
            banner.textContent = message;
            banner.classList.add('show');
            setTimeout(() => banner.classList.remove('show'), 5000);
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

        function sendMotor(linear, angular) {
            fetch('/api/motor', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({linear: linear, angular: angular})
            });
        }

        function sendStop() {
            sendMotor(0, 0);
        }

        function emergencyStop() {
            fetch('/api/motor', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({emergency: true})
            });
            showAlert('EMERGENCY STOP ACTIVATED!');
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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
