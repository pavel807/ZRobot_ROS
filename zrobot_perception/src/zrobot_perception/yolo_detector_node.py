#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 Jazzy node for YOLO-based object detection using RKNN NPU.
Replaces the C++ yolo_detector_node with a Python multiprocessing implementation.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, ObjectHypothesis
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
import multiprocessing as mp
import os
import time
import threading
from collections import deque

# --- COCO Classes (80) ---
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
    78: 'hair drier', 79: 'toothbrush'
}

# Reverse lookup
COCO_CLASSES_REV = {v: k for k, v in COCO_CLASSES.items()}


# --- NPU Process ---
def npu_process(core_id, input_queue, output_queue, model_path, obj_thresh, nms_thresh):
    """
    Отдельный ПРОЦЕСС для каждого NPU ядра.
    Импорт RKNN делаем здесь, чтобы избежать проблем с fork.
    """
    from rknnlite.api import RKNNLite

    # Привязываем процесс к конкретному CPU ядру для стабильности (RK3588 big cores: 4,5,6,7)
    try:
        os.sched_setaffinity(0, {4 + core_id})
    except Exception:
        pass  # Fallback if affinity fails

    rknn = RKNNLite()
    rknn.load_rknn(model_path)

    core_masks = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]
    rknn.init_runtime(core_mask=core_masks[core_id])

    print(f"[NPU Core {core_id}] Started")

    while True:
        task = input_queue.get()
        if task is None:
            break

        frame_id, rgb_frame = task

        # Inference
        outputs = rknn.inference(inputs=[np.expand_dims(rgb_frame, 0)])
        out = np.squeeze(outputs[0])

        # Post-processing
        boxes_raw = out[:4, :].transpose()
        probs = out[4:, :].transpose()
        confidences = np.max(probs, axis=1)
        class_ids = np.argmax(probs, axis=1)

        mask = confidences > obj_thresh
        conf = confidences[mask]
        c_ids = class_ids[mask]
        b_raw = boxes_raw[mask]

        detections = []
        if len(conf) > 0:
            boxes = np.empty_like(b_raw)
            boxes[:, 0] = b_raw[:, 0] - b_raw[:, 2] / 2
            boxes[:, 1] = b_raw[:, 1] - b_raw[:, 3] / 2
            boxes[:, 2] = b_raw[:, 2]
            boxes[:, 3] = b_raw[:, 3]

            indices = cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(), obj_thresh, nms_thresh)
            if len(indices) > 0:
                indices = indices.flatten()
                detections = [(boxes[i].tolist(), float(conf[i]), int(c_ids[i])) for i in indices]

        output_queue.put((frame_id, rgb_frame, detections))


# --- Capture Process ---
def capture_process(camera_id, width, height, fps, prep_queue):
    """Процесс захвата с максимальной скоростью"""
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    if not cap.isOpened():
        print("[Capture] Failed to open camera")
        return

    print(f"[Capture] Started: {width}x{height}@{fps}")
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Preprocessing
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if prep_queue.full():
            try:
                prep_queue.get_nowait()
            except Exception:
                pass
        prep_queue.put((frame_id, rgb))
        frame_id += 1


# ============================================================================
# YOLO Detector ROS2 Node
# ============================================================================
class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        self.get_logger().info('Initializing YOLO Detector Node (RKNN NPU)...')

        # --- Declare parameters ---
        self.declare_parameter('model_path', 'models/yolov8n.rknn')
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 640)
        self.declare_parameter('fps', 30)
        self.declare_parameter('obj_thresh', 0.25)
        self.declare_parameter('nms_thresh', 0.45)
        self.declare_parameter('num_npu_cores', 3)
        self.declare_parameter('target_object', 'person')
        self.declare_parameter('enable_tracking', True)
        self.declare_parameter('show_category', True)
        self.declare_parameter('enable_auto_follow', True)
        self.declare_parameter('max_linear_speed', 0.3)
        self.declare_parameter('turn_speed', 0.5)

        # --- Get parameters ---
        self.model_path = self.get_parameter('model_path').value
        self.camera_id = self.get_parameter('camera_id').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.fps = self.get_parameter('fps').value
        self.obj_thresh = self.get_parameter('obj_thresh').value
        self.nms_thresh = self.get_parameter('nms_thresh').value
        self.num_npu_cores = self.get_parameter('num_npu_cores').value
        self.target_object = self.get_parameter('target_object').value
        self.enable_tracking = self.get_parameter('enable_tracking').value
        self.show_category = self.get_parameter('show_category').value
        self.enable_auto_follow = self.get_parameter('enable_auto_follow').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.turn_speed = self.get_parameter('turn_speed').value

        self.get_logger().info(f'Model: {self.model_path}')
        self.get_logger().info(f'Camera: {self.camera_id}, {self.width}x{self.height}@{self.fps}')
        self.get_logger().info(f'Thresholds: obj={self.obj_thresh}, nms={self.nms_thresh}')
        self.get_logger().info(f'NPU Cores: {self.num_npu_cores}')
        self.get_logger().info(f'Target: {self.target_object}')

        # --- Publishers ---
        self.detections_pub = self.create_publisher(Detection2DArray, 'detections', 10)
        self.image_pub = self.create_publisher(Image, 'processed_image', 10)
        self.status_pub = self.create_publisher(String, 'detection_status', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # --- Subscriptions ---
        self.target_sub = self.create_subscription(
            String, 'set_target', self.target_callback, 10)
        self.confidence_sub = self.create_subscription(
            Float32, 'set_confidence', self.confidence_callback, 10)

        # --- Queues for multiprocessing ---
        self.prep_queue = mp.Queue(maxsize=6)
        self.result_queue = mp.Queue(maxsize=6)

        # --- Launch NPU processes ---
        self.npu_processes = []
        for i in range(self.num_npu_cores):
            p = mp.Process(
                target=npu_process,
                args=(i, self.prep_queue, self.result_queue,
                      self.model_path, self.obj_thresh, self.nms_thresh))
            p.start()
            self.npu_processes.append(p)

        # --- Launch capture process ---
        self.cap_proc = mp.Process(
            target=capture_process,
            args=(self.camera_id, self.width, self.height, self.fps, self.prep_queue))
        self.cap_proc.start()

        # --- State ---
        self.cv_bridge = CvBridge()
        self.current_frame = None
        self.current_detections = []
        self.frame_lock = threading.Lock()

        # FPS / performance tracking
        self.frame_count = 0
        self.last_fps_time = self.get_clock().now()
        self.current_fps = 0.0
        self.inference_times = deque(maxlen=60)

        # Tracking state
        self.tracker_history = {}
        self.next_track_id = 0

        # Auto-follow state
        self.last_detection_time = time.time()
        self.search_start_time = time.time()
        self.in_search_mode = False
        self.search_direction = 1
        self.speed_history_l = deque(maxlen=4)
        self.speed_history_r = deque(maxlen=4)

        # Motor control params (mirrored from C++ node)
        self.max_speed = 245
        self.min_speed = 165
        self.tracking_speed = 220
        self.search_speed = 115
        self.center_threshold = 0.08
        self.max_turn_ratio = 0.95
        self.approach_width = 120

        # --- Timer for result collection ---
        self.timer = self.create_timer(0.01, self.collect_results)  # 100 Hz

        # --- Timer for detection publishing ---
        self.publish_timer = self.create_timer(1.0 / self.fps, self.publish_detections)

        self.get_logger().info('YOLO Detector Node started')

    def target_callback(self, msg: String):
        if self.target_object != msg.data:
            self.get_logger().info(f'Target changed to: {msg.data}')
            self.target_object = msg.data
            self.tracker_history.clear()
            self.in_search_mode = False
            self.search_direction = 1

    def confidence_callback(self, msg: Float32):
        if 0.0 <= msg.data <= 1.0:
            old = self.obj_thresh
            self.obj_thresh = msg.data
            self.get_logger().info(f'Confidence threshold: {old:.2f} -> {self.obj_thresh:.2f}')
        else:
            self.get_logger().warn(f'Invalid confidence: {msg.data}')

    def collect_results(self):
        """Collect results from NPU processes and update current frame"""
        try:
            while not self.result_queue.empty():
                frame_id, rgb_frame, detections = self.result_queue.get_nowait()

                t_start = time.time()

                # Convert to BGR for display
                frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

                # Draw detections
                for (box, score, cls_id) in detections:
                    x1, y1 = int(box[0]), int(box[1])
                    x2, y2 = int(box[0] + box[2]), int(box[1] + box[3])
                    class_name = COCO_CLASSES.get(cls_id, str(cls_id))
                    label = f"{class_name} {score:.2f}"

                    # Color by category
                    color = self.get_category_color(class_name)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw FPS
                cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Target: {self.target_object}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                mode_text = "SEARCHING" if self.in_search_mode else "TRACKING" if detections else "IDLE"
                mode_color = (0, 165, 255) if self.in_search_mode else (0, 255, 0) if detections else (128, 128, 128)
                cv2.putText(frame, mode_text, (frame.shape[1] - 140, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

                with self.frame_lock:
                    self.current_frame = frame
                    self.current_detections = detections
                    self.last_detection_time = time.time()

                t_end = time.time()
                self.inference_times.append((t_end - t_start) * 1000)

        except Exception:
            pass

    def publish_detections(self):
        """Publish detection messages and processed images"""
        # Update FPS
        self.frame_count += 1
        now = self.get_clock().now()
        elapsed = (now - self.last_fps_time).nanoseconds / 1e9
        if elapsed >= 0.5:
            self.current_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_time = now

        with self.frame_lock:
            if self.current_frame is None:
                return
            frame = self.current_frame.copy()
            detections = self.current_detections.copy()

        # Publish processed image
        if self.image_pub.get_subscription_count() > 0:
            header = rclpy.time.Time().to_msg()
            header.stamp = self.get_clock().now().to_msg()
            img_msg = self.cv_bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            img_msg.header.stamp = header.stamp
            self.image_pub.publish(img_msg)

        # Publish detections array
        if self.detections_pub.get_subscription_count() > 0:
            det_array = Detection2DArray()
            det_array.header.stamp = self.get_clock().now().to_msg()
            det_array.header.frame_id = 'camera'

            for (box, score, cls_id) in detections:
                det = Detection2D()
                det.header = det_array.header

                # Bounding box
                det.bbox.center.position.x = box[0] + box[2] / 2
                det.bbox.center.position.y = box[1] + box[3] / 2
                det.bbox.size_x = float(box[2])
                det.bbox.size_y = float(box[3])

                # Object hypothesis
                hypothesis = ObjectHypothesis()
                hypothesis.class_id = COCO_CLASSES.get(cls_id, str(cls_id))
                hypothesis.score = score
                hyp_with_pose = ObjectHypothesisWithPose()
                hyp_with_pose.hypothesis = hypothesis
                det.results.append(hyp_with_pose)

                det_array.detections.append(det)

            self.detections_pub.publish(det_array)

        # Publish status
        if self.status_pub.get_subscription_count() > 0:
            status = String()
            status.data = f"FPS:{self.current_fps:.1f}|Dets:{len(detections)}|Target:{self.target_object}"
            self.status_pub.publish(status)

        # Auto-follow control
        if self.enable_auto_follow and self.cmd_vel_pub.get_subscription_count() > 0:
            self.publish_cmd_vel(detections, frame.shape[1])

    def publish_cmd_vel(self, detections, frame_width):
        """Publish velocity commands for auto-follow"""
        twist = Twist()
        found = len(detections) > 0

        if not found:
            # Search mode
            if not self.in_search_mode:
                self.in_search_mode = True
                self.search_start_time = time.time()
                self.search_direction = 1

            elapsed = time.time() - self.search_start_time
            if elapsed > 2.0:  # Switch direction every 2 seconds
                self.search_direction *= -1
                self.search_start_time = time.time()

            if self.search_direction > 0:
                twist.angular.z = -self.turn_speed * 0.5
            else:
                twist.angular.z = self.turn_speed * 0.5
        else:
            self.in_search_mode = False

            # Find target detection
            target_det = None
            for (box, score, cls_id) in detections:
                class_name = COCO_CLASSES.get(cls_id, '')
                if class_name == self.target_object:
                    target_det = (box, score, cls_id)
                    break

            if target_det is None and len(detections) > 0:
                target_det = detections[0]  # Fallback to first detection

            if target_det is not None:
                box, score, cls_id = target_det
                center_x = box[0] + box[2] / 2
                normalized_center = (center_x / frame_width) - 0.5

                # Calculate turn speed proportional to deviation
                turn = normalized_center * self.turn_speed
                turn = max(-self.turn_speed, min(self.turn_speed, turn))

                # Calculate approach speed based on object size
                obj_width = box[2]
                if obj_width > self.approach_width:
                    distance_factor = (obj_width - self.approach_width) / 300.0
                    distance_factor = min(1.0, distance_factor)
                    scale = 1.0 - (distance_factor * 0.7)
                    scale = max(0.25, scale)
                    forward = self.max_linear_speed * scale
                else:
                    forward = self.max_linear_speed

                # Apply center threshold
                if abs(normalized_center) < self.center_threshold:
                    twist.angular.z = 0.0
                    twist.linear.x = forward
                else:
                    twist.angular.z = -turn  # Negative because camera is inverted
                    twist.linear.x = forward * 0.5  # Slow down while turning

        self.cmd_vel_pub.publish(twist)

    def get_category_color(self, class_name: str):
        """Get BGR color for category"""
        # Simplified category colors
        person = (0, 0, 255)        # Red
        vehicle = (0, 255, 0)       # Green
        animal = (255, 0, 0)        # Blue
        food = (0, 255, 255)        # Yellow
        electronic = (255, 0, 255)  # Magenta
        default = (255, 255, 0)     # Cyan

        if class_name == 'person':
            return person
        if class_name in ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']:
            return vehicle
        if class_name in ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                          'zebra', 'giraffe']:
            return animal
        if class_name in ['apple', 'orange', 'banana', 'broccoli', 'carrot', 'sandwich', 'pizza',
                          'donut', 'cake', 'hot dog', 'bowl', 'cup', 'wine glass', 'bottle']:
            return food
        if class_name in ['tv', 'laptop', 'mouse', 'keyboard', 'cell phone', 'remote', 'microwave',
                          'oven', 'toaster', 'refrigerator']:
            return electronic
        return default

    def destroy_node(self):
        """Cleanup on shutdown"""
        self.get_logger().info('Shutting down YOLO Detector Node...')

        # Stop NPU processes
        for _ in self.npu_processes:
            try:
                self.prep_queue.put_nowait(None)
            except Exception:
                pass

        for p in self.npu_processes:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()

        # Stop capture process
        if self.cap_proc.is_alive():
            self.cap_proc.terminate()
            self.cap_proc.join(timeout=2)

        super().destroy_node()


def main(args=None):
    mp.set_start_method('spawn', force=True)  # Critical for RKNN!
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
