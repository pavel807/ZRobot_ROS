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
from aiohttp import web
import base64
from cv_bridge import CvBridge
import argparse


class WebInterfaceNode(Node):
    def __init__(self):
        super().__init__('web_interface')

        # Declare parameters
        self.declare_parameter('host', '0.0.0.0')
        self.declare_parameter('port', 8080)

        self.host = self.get_parameter('host').get_parameter_value().string_value
        self.port = self.get_parameter('port').get_parameter_value().integer_value

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            'processed_image',
            self.image_callback,
            10
        )

        self.status_sub = self.create_subscription(
            String,
            'detection_status',
            self.status_callback,
            10
        )

        self.detections_sub = self.create_subscription(
            Detection2DArray,
            'detections',
            self.detections_callback,
            10
        )

        self.tracked_sub = self.create_subscription(
            String,
            'tracked_objects',
            self.tracked_callback,
            10
        )

        # Publishers for settings
        self.target_pub = self.create_publisher(String, 'set_target', 10)
        self.confidence_pub = self.create_publisher(Float32, 'set_confidence', 10)
        self.motor_enable_pub = self.create_publisher(Bool, 'motor_enable', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # State
        self.current_frame = None
        self.current_status = {
            'target': '-',
            'found': False,
            'zone': 'NONE',
            'count': 0,
            'inference_time': 0,
            'fps': 0,
            'tracked': 0,
            'classes': ''
        }
        self.current_detections = []
        self.current_tracked = []
        self.websocket_clients = set()

        # Settings state
        self.settings = {
            'target_object': 'person',
            'conf_threshold': 0.25,
            'camera_id': 1,
            'enable_tracking': True,
            'adaptive_threshold': True,
            'show_category': True,
            'motor_enabled': True,
            'max_speed': 245
        }

        self.bridge = CvBridge()
        self.frame_lock = threading.Lock()

        self.get_logger().info('Web Interface Node initialized')

    def image_callback(self, msg):
        with self.frame_lock:
            try:
                self.current_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            except Exception as e:
                self.get_logger().error(f'Image conversion error: {e}')

    def status_callback(self, msg):
        try:
            self.current_status = json.loads(msg.data)
        except:
            pass

    def detections_callback(self, msg):
        self.current_detections = []
        for det in msg.detections:
            self.current_detections.append({
                'class': det.results[0].hypothesis.class_id,
                'score': det.results[0].hypothesis.score,
                'bbox': {
                    'x': det.bbox.center.position.x,
                    'y': det.bbox.center.position.y,
                    'width': det.bbox.size_x,
                    'height': det.bbox.size_y
                }
            })

    def tracked_callback(self, msg):
        try:
            self.current_tracked = json.loads(msg.data)
        except:
            pass

    def broadcast_status(self):
        if self.websocket_clients:
            asyncio.new_event_loop().run_until_complete(
                self.send_to_all(json.dumps({
                    'type': 'status',
                    'data': self.current_status,
                    'tracked': self.current_tracked,
                    'settings': self.settings
                }))
            )

    async def send_to_all(self, message):
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
        app.router.add_post('/api/settings', self.handle_settings)
        app.router.add_post('/api/target', self.handle_set_target)
        app.router.add_post('/api/confidence', self.handle_set_confidence)
        app.router.add_post('/api/motor', self.handle_motor_control)
        app.router.add_post('/api/speed', self.handle_set_speed)
        app.router.add_get('/ws', self.handle_websocket)

        web.run_app(app, host=self.host, port=self.port, handle_signals=False)

    async def handle_index(self, request):
        content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 ZRobot Control Center</title>
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

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, var(--bg-dark) 0%, #1a1a3e 100%);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }

        .header {
            background: linear-gradient(90deg, var(--bg-card) 0%, var(--bg-panel) 100%);
            padding: 20px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid var(--primary);
            box-shadow: 0 4px 20px rgba(0, 217, 255, 0.2);
        }

        .header h1 {
            font-size: 28px;
            background: linear-gradient(90deg, var(--primary), var(--success));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .header-status {
            display: flex;
            gap: 20px;
            align-items: center;
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: var(--bg-dark);
            border-radius: 20px;
            font-size: 14px;
        }

        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        .status-dot.connected { background: var(--success); }
        .status-dot.disconnected { background: var(--danger); }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .main-container {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 20px;
            padding: 20px 40px;
            max-width: 1800px;
            margin: 0 auto;
        }

        .video-section {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .video-container {
            background: var(--bg-card);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(0, 217, 255, 0.2);
        }

        #videoCanvas {
            width: 100%;
            display: block;
            background: #000;
        }

        .stats-bar {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 15px;
            padding: 20px;
            background: var(--bg-panel);
            border-radius: 12px;
        }

        .stat-item {
            text-align: center;
            padding: 15px;
            background: linear-gradient(135deg, rgba(0, 217, 255, 0.1), rgba(123, 44, 191, 0.1));
            border-radius: 10px;
            border: 1px solid rgba(0, 217, 255, 0.2);
            transition: transform 0.3s, box-shadow 0.3s;
        }

        .stat-item:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(0, 217, 255, 0.3);
        }

        .stat-label {
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }

        .stat-value {
            font-size: 24px;
            font-weight: bold;
            background: linear-gradient(90deg, var(--primary), var(--success));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .panel {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(0, 217, 255, 0.2);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }

        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid rgba(0, 217, 255, 0.2);
        }

        .panel-title {
            font-size: 18px;
            font-weight: 600;
            color: var(--primary);
        }

        .settings-group {
            margin-bottom: 20px;
        }

        .settings-group label {
            display: block;
            font-size: 13px;
            color: var(--text-secondary);
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .settings-group select,
        .settings-group input[type="range"] {
            width: 100%;
            padding: 12px;
            background: var(--bg-dark);
            border: 1px solid rgba(0, 217, 255, 0.3);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 14px;
            transition: border-color 0.3s;
        }

        .settings-group select:focus,
        .settings-group input:focus {
            outline: none;
            border-color: var(--primary);
        }

        .settings-group input[type="range"] {
            padding: 0;
            height: 8px;
            -webkit-appearance: none;
            appearance: none;
        }

        .settings-group input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            background: var(--primary);
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 2px 10px rgba(0, 217, 255, 0.5);
        }

        .range-value {
            text-align: right;
            font-size: 14px;
            color: var(--primary);
            margin-top: 5px;
        }

        .toggle-group {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
        }

        .toggle-label {
            font-size: 14px;
            color: var(--text-primary);
        }

        .toggle-switch {
            position: relative;
            width: 50px;
            height: 26px;
        }

        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }

        .toggle-slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: var(--bg-dark);
            border-radius: 26px;
            transition: 0.3s;
            border: 1px solid rgba(0, 217, 255, 0.3);
        }

        .toggle-slider:before {
            position: absolute;
            content: "";
            height: 18px;
            width: 18px;
            left: 3px;
            bottom: 3px;
            background: var(--text-secondary);
            border-radius: 50%;
            transition: 0.3s;
        }

        .toggle-switch input:checked + .toggle-slider {
            background: var(--primary);
            border-color: var(--primary);
        }

        .toggle-switch input:checked + .toggle-slider:before {
            transform: translateX(24px);
            background: #fff;
        }

        .btn {
            width: 100%;
            padding: 14px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .btn-primary {
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            color: #fff;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 217, 255, 0.4);
        }

        .btn-danger {
            background: linear-gradient(90deg, var(--danger), #ff0044);
            color: #fff;
        }

        .btn-success {
            background: linear-gradient(90deg, var(--success), #00cc66);
            color: #fff;
        }

        .detections-list {
            max-height: 200px;
            overflow-y: auto;
        }

        .detection-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            background: var(--bg-dark);
            border-radius: 6px;
            margin-bottom: 8px;
            border-left: 3px solid var(--primary);
        }

        .detection-class {
            font-weight: 600;
            color: var(--primary);
        }

        .detection-score {
            background: linear-gradient(90deg, var(--primary), var(--success));
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }

        .motor-controls {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 15px;
        }

        .motor-btn {
            padding: 20px;
            font-size: 24px;
            background: var(--bg-dark);
            border: 2px solid var(--primary);
            border-radius: 10px;
            color: var(--primary);
            cursor: pointer;
            transition: all 0.2s;
        }

        .motor-btn:hover, .motor-btn:active {
            background: var(--primary);
            color: var(--bg-dark);
        }

        .motor-btn.stop {
            border-color: var(--danger);
            color: var(--danger);
        }

        .motor-btn.stop:hover {
            background: var(--danger);
            color: #fff;
        }

        .tracked-objects {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }

        .tracked-badge {
            background: linear-gradient(135deg, var(--secondary), var(--primary));
            padding: 6px 12px;
            border-radius: 16px;
            font-size: 12px;
            font-weight: 600;
        }

        @media (max-width: 1200px) {
            .main-container {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <header class="header">
        <h1>🤖 ZRobot Control Center</h1>
        <div class="header-status">
            <div class="status-indicator">
                <span class="status-dot" id="wsStatus"></span>
                <span id="wsStatusText">Connecting...</span>
            </div>
        </div>
    </header>

    <div class="main-container">
        <div class="video-section">
            <div class="video-container">
                <canvas id="videoCanvas" width="640" height="480"></canvas>
            </div>

            <div class="stats-bar">
                <div class="stat-item">
                    <div class="stat-label">Target</div>
                    <div class="stat-value" id="targetDisplay">-</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Status</div>
                    <div class="stat-value" id="foundDisplay">-</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Zone</div>
                    <div class="stat-value" id="zoneDisplay">-</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">FPS</div>
                    <div class="stat-value" id="fpsDisplay">0</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Latency</div>
                    <div class="stat-value" id="latencyDisplay">0ms</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Tracked</div>
                    <div class="stat-value" id="trackedDisplay">0</div>
                </div>
            </div>
        </div>

        <div class="sidebar">
            <div class="panel">
                <div class="panel-header">
                    <span class="panel-title">⚙️ Detection Settings</span>
                </div>

                <div class="settings-group">
                    <label>Target Object</label>
                    <select id="targetSelect" onchange="updateSetting('target_object', this.value)">
                        <optgroup label="Person">
                            <option value="person">👤 Person</option>
                        </optgroup>
                        <optgroup label="Vehicle">
                            <option value="bicycle">🚴 Bicycle</option>
                            <option value="car">🚗 Car</option>
                            <option value="motorcycle">🏍️ Motorcycle</option>
                            <option value="airplane">✈️ Airplane</option>
                            <option value="bus">🚌 Bus</option>
                            <option value="train">🚂 Train</option>
                            <option value="truck">🚚 Truck</option>
                            <option value="boat">⛵ Boat</option>
                        </optgroup>
                        <optgroup label="Outdoor">
                            <option value="traffic light">🚦 Traffic Light</option>
                            <option value="fire hydrant">🧯 Fire Hydrant</option>
                            <option value="stop sign">🛑 Stop Sign</option>
                            <option value="parking meter">🅿️ Parking Meter</option>
                            <option value="bench">🪑 Bench</option>
                        </optgroup>
                        <optgroup label="Animal">
                            <option value="bird">🐦 Bird</option>
                            <option value="cat">🐱 Cat</option>
                            <option value="dog">🐶 Dog</option>
                            <option value="horse">🐴 Horse</option>
                            <option value="sheep">🐑 Sheep</option>
                            <option value="cow">🐄 Cow</option>
                            <option value="elephant">🐘 Elephant</option>
                            <option value="bear">🐻 Bear</option>
                            <option value="zebra">🦓 Zebra</option>
                            <option value="giraffe">🦒 Giraffe</option>
                        </optgroup>
                        <optgroup label="Accessory">
                            <option value="backpack">🎒 Backpack</option>
                            <option value="umbrella">☂️ Umbrella</option>
                            <option value="handbag">👜 Handbag</option>
                            <option value="tie">👔 Tie</option>
                            <option value="suitcase">💼 Suitcase</option>
                        </optgroup>
                        <optgroup label="Sports">
                            <option value="frisbee">🥏 Frisbee</option>
                            <option value="skis">⛷️ Skis</option>
                            <option value="snowboard">🏂 Snowboard</option>
                            <option value="sports ball">⚽ Sports Ball</option>
                            <option value="kite">🪁 Kite</option>
                            <option value="baseball bat">⚾ Baseball Bat</option>
                            <option value="baseball glove">🧤 Baseball Glove</option>
                            <option value="skateboard">🛹 Skateboard</option>
                            <option value="surfboard">🏄 Surfboard</option>
                            <option value="tennis racket">🎾 Tennis Racket</option>
                        </optgroup>
                        <optgroup label="Food & Drink">
                            <option value="bottle">🍶 Bottle</option>
                            <option value="wine glass">🍷 Wine Glass</option>
                            <option value="cup">☕ Cup</option>
                            <option value="fork">🍴 Fork</option>
                            <option value="knife">🔪 Knife</option>
                            <option value="spoon">🥄 Spoon</option>
                            <option value="bowl">🥣 Bowl</option>
                            <option value="banana">🍌 Banana</option>
                            <option value="apple">🍎 Apple</option>
                            <option value="sandwich">🥪 Sandwich</option>
                            <option value="orange">🍊 Orange</option>
                            <option value="broccoli">🥦 Broccoli</option>
                            <option value="carrot">🥕 Carrot</option>
                            <option value="hot dog">🌭 Hot Dog</option>
                            <option value="pizza">🍕 Pizza</option>
                            <option value="donut">🍩 Donut</option>
                            <option value="cake">🎂 Cake</option>
                        </optgroup>
                        <optgroup label="Furniture">
                            <option value="chair">🪑 Chair</option>
                            <option value="couch">🛋️ Couch</option>
                            <option value="potted plant">🪴 Potted Plant</option>
                            <option value="bed">🛏️ Bed</option>
                            <option value="dining table">🍽️ Dining Table</option>
                            <option value="toilet">🚽 Toilet</option>
                        </optgroup>
                        <optgroup label="Electronics">
                            <option value="tv">📺 TV</option>
                            <option value="laptop">💻 Laptop</option>
                            <option value="mouse">🖱️ Mouse</option>
                            <option value="remote">📱 Remote</option>
                            <option value="keyboard">⌨️ Keyboard</option>
                            <option value="cell phone">📱 Cell Phone</option>
                        </optgroup>
                        <optgroup label="Appliances">
                            <option value="microwave">🔬 Microwave</option>
                            <option value="oven">🔥 Oven</option>
                            <option value="toaster">🍞 Toaster</option>
                            <option value="sink">🚰 Sink</option>
                            <option value="refrigerator">🧊 Refrigerator</option>
                        </optgroup>
                        <optgroup label="Indoor">
                            <option value="book">📖 Book</option>
                            <option value="clock">🕐 Clock</option>
                            <option value="vase">🏺 Vase</option>
                            <option value="scissors">✂️ Scissors</option>
                            <option value="teddy bear">🧸 Teddy Bear</option>
                            <option value="hair drier">💨 Hair Drier</option>
                            <option value="toothbrush">🪥 Toothbrush</option>
                        </optgroup>
                    </select>
                </div>

                <div class="settings-group">
                    <label>Confidence Threshold: <span id="confValue">0.25</span></label>
                    <input type="range" id="confSlider" min="0.1" max="0.8" step="0.05" 
                           value="0.25" onchange="updateConfidence(this.value)">
                </div>

                <div class="toggle-group">
                    <span class="toggle-label">🎯 Object Tracking</span>
                    <label class="toggle-switch">
                        <input type="checkbox" id="trackingToggle" checked onchange="toggleSetting('enable_tracking', this.checked)">
                        <span class="toggle-slider"></span>
                    </label>
                </div>

                <div class="toggle-group">
                    <span class="toggle-label">📊 Adaptive Threshold</span>
                    <label class="toggle-switch">
                        <input type="checkbox" id="adaptiveToggle" checked onchange="toggleSetting('adaptive_threshold', this.checked)">
                        <span class="toggle-slider"></span>
                    </label>
                </div>

                <div class="toggle-group">
                    <span class="toggle-label">🏷️ Show Categories</span>
                    <label class="toggle-switch">
                        <input type="checkbox" id="categoryToggle" checked onchange="toggleSetting('show_category', this.checked)">
                        <span class="toggle-slider"></span>
                    </label>
                </div>
            </div>

            <div class="panel">
                <div class="panel-header">
                    <span class="panel-title">🔍 Detected Objects</span>
                </div>
                <div class="detections-list" id="detectionsList">
                    <div style="color: var(--text-secondary); text-align: center; padding: 20px;">
                        No detections yet
                    </div>
                </div>
                <div class="tracked-objects" id="trackedObjects"></div>
            </div>

            <div class="panel">
                <div class="panel-header">
                    <span class="panel-title">🎮 Motor Control</span>
                </div>
                <div class="motor-controls">
                    <button class="motor-btn" onmousedown="sendMotor(0, 0.5)" onmouseup="sendStop()" ontouchstart="sendMotor(0, 0.5)" ontouchend="sendStop()">⬆️</button>
                    <button class="motor-btn stop" onclick="sendStop()">⏹️</button>
                    <button class="motor-btn" onmousedown="sendMotor(0, -0.5)" onmouseup="sendStop()" ontouchstart="sendMotor(0, -0.5)" ontouchend="sendStop()">⬇️</button>
                    <button class="motor-btn" onmousedown="sendMotor(-0.3, 0)" onmouseup="sendStop()" ontouchstart="sendMotor(-0.3, 0)" ontouchend="sendStop()">⬅️</button>
                    <button class="motor-btn" style="visibility:hidden"></button>
                    <button class="motor-btn" onmousedown="sendMotor(0.3, 0)" onmouseup="sendStop()" ontouchstart="sendMotor(0.3, 0)" ontouchend="sendStop()">➡️</button>
                </div>
                <button class="btn btn-danger" style="margin-top: 15px;" onclick="toggleMotor()">
                    🛑 Emergency Stop
                </button>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let lastFrameTime = 0;

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(protocol + '//' + window.location.host + '/ws');

            ws.onopen = function() {
                document.getElementById('wsStatus').className = 'status-dot connected';
                document.getElementById('wsStatusText').textContent = 'Connected';
            };

            ws.onclose = function() {
                document.getElementById('wsStatus').className = 'status-dot disconnected';
                document.getElementById('wsStatusText').textContent = 'Disconnected';
                setTimeout(connectWebSocket, 2000);
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'status') {
                    updateStatus(data.data);
                    if (data.settings) {
                        updateSettingsUI(data.settings);
                    }
                }
            };

            ws.onerror = function() {
                ws.close();
            };
        }

        function updateStatus(data) {
            document.getElementById('targetDisplay').textContent = data.target || '-';
            document.getElementById('foundDisplay').textContent = data.found ? '✅ YES' : '❌ NO';
            document.getElementById('foundDisplay').style.background = data.found ? 
                'linear-gradient(90deg, #00ff88, #00cc66)' : 
                'linear-gradient(90deg, #ff4466, #ff0044)';
            document.getElementById('foundDisplay').style.webkitBackgroundClip = 'text';
            document.getElementById('zoneDisplay').textContent = data.zone || 'NONE';
            document.getElementById('fpsDisplay').textContent = (data.fps || 0).toFixed(1);
            document.getElementById('latencyDisplay').textContent = (data.inference_time || 0).toFixed(1) + 'ms';
            document.getElementById('trackedDisplay').textContent = data.tracked || 0;

            // Update detections list
            const detectionsList = document.getElementById('detectionsList');
            if (data.classes) {
                const classes = data.classes.split(', ');
                detectionsList.innerHTML = classes.map(cls => `
                    <div class="detection-item">
                        <span class="detection-class">${cls}</span>
                        <span class="detection-score">Detected</span>
                    </div>
                `).join('');
            }

            // Update tracked objects badges
            const trackedContainer = document.getElementById('trackedObjects');
            if (data.tracked > 0) {
                trackedContainer.innerHTML = Array(data.tracked).fill(0).map((_, i) => 
                    `<span class="tracked-badge">#${i}</span>`
                ).join('');
            } else {
                trackedContainer.innerHTML = '';
            }
        }

        function updateSettingsUI(settings) {
            document.getElementById('targetSelect').value = settings.target_object || 'person';
            document.getElementById('confSlider').value = settings.conf_threshold || 0.25;
            document.getElementById('confValue').textContent = settings.conf_threshold || 0.25;
            document.getElementById('trackingToggle').checked = settings.enable_tracking !== false;
            document.getElementById('adaptiveToggle').checked = settings.adaptive_threshold !== false;
            document.getElementById('categoryToggle').checked = settings.show_category !== false;
        }

        function updateSetting(key, value) {
            fetch('/api/settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({[key]: value})
            });
        }

        function updateConfidence(value) {
            document.getElementById('confValue').textContent = value;
            fetch('/api/confidence', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({confidence: parseFloat(value)})
            });
        }

        function toggleSetting(key, value) {
            fetch('/api/settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({[key]: value})
            });
        }

        function sendMotor(linear, angular) {
            fetch('/api/motor', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({linear: linear, angular: angular})
            });
        }

        function sendStop() {
            sendMotor(0, 0);
        }

        function toggleMotor() {
            fetch('/api/motor', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({stop: true})
            });
        }

        // Video streaming
        function updateVideo() {
            const canvas = document.getElementById('videoCanvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();
            img.onload = function() {
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            };
            img.src = '/api/frame?t=' + Date.now();
            setTimeout(updateVideo, 100);
        }

        // Initialize
        connectWebSocket();
        updateVideo();
    </script>
</body>
</html>
"""
        return web.Response(text=content, content_type='text/html')

    async def handle_frame(self, request):
        with self.frame_lock:
            if self.current_frame is None:
                return web.Response(status=204)
            _, buffer = cv2.imencode('.jpg', self.current_frame, 
                                      [cv2.IMWRITE_JPEG_QUALITY, 80])
            return web.Response(body=buffer.tobytes(), content_type='image/jpeg')

    async def handle_status(self, request):
        return web.json_response(self.current_status)

    async def handle_settings(self, request):
        data = await request.json()
        for key, value in data.items():
            if key in self.settings:
                self.settings[key] = value
                self.get_logger().info(f'Setting updated: {key} = {value}')
                
                # Publish to ROS2 topics
                if key == 'target_object':
                    msg = String()
                    msg.data = str(value)
                    self.target_pub.publish(msg)
        
        return web.json_response({'success': True, 'settings': self.settings})

    async def handle_set_target(self, request):
        data = await request.json()
        target = data.get('target', 'person')
        
        msg = String()
        msg.data = target
        self.target_pub.publish(msg)
        self.settings['target_object'] = target
        
        return web.json_response({'success': True, 'target': target})

    async def handle_set_confidence(self, request):
        data = await request.json()
        confidence = data.get('confidence', 0.25)
        
        msg = Float32()
        msg.data = float(confidence)
        self.confidence_pub.publish(msg)
        self.settings['conf_threshold'] = float(confidence)
        
        return web.json_response({'success': True, 'confidence': confidence})

    async def handle_motor_control(self, request):
        data = await request.json()

        if data.get('stop'):
            # Stop motors
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            return web.json_response({'success': True, 'action': 'stop'})

        linear = data.get('linear', 0)
        angular = data.get('angular', 0)

        # Publish Twist message for motor control
        twist = Twist()
        twist.linear.x = float(linear)
        twist.angular.z = float(angular)
        self.cmd_vel_pub.publish(twist)

        return web.json_response({'success': True, 'linear': linear, 'angular': angular})

    async def handle_set_speed(self, request):
        data = await request.json()
        speed = data.get('speed', 245)
        self.settings['max_speed'] = int(speed)
        return web.json_response({'success': True, 'speed': speed})

    async def handle_websocket(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.websocket_clients.add(ws)

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    await ws.send_str(json.dumps({
                        'type': 'ack',
                        'settings': self.settings
                    }))
        finally:
            self.websocket_clients.remove(ws)

        return ws


def main():
    parser = argparse.ArgumentParser(description='ZRobot Web Interface')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    args, unknown = parser.parse_known_args()

    rclpy.init(args=unknown)
    node = WebInterfaceNode()

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
