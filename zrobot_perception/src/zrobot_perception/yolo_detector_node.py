#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced YOLO Detector Node — multiprocessing NPU inference (RK3588).
СТАБИЛЬНАЯ ВЕРСИЯ:
  ✅ Надёжный shutdown через mp.Event + таймауты очередей
  ✅ Обработка переполнения очередей без блокировок
  ✅ Корректное масштабирование координат 640×640 → 640×480
  ✅ Безопасная передача данных (только Python-типы)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, ObjectHypothesis
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
import multiprocessing as mp
import time
import os
import json
import threading
import signal
import sys

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

IMG_SIZE = 640
QUEUE_TIMEOUT = 0.1  # секунды — предотвращает вечные блокировки


def npu_process(core_id, input_queue, output_queue, model_path, obj_thresh, nms_thresh, shutdown_event):
    """
    NPU worker process с надёжной обработкой shutdown.
    """
    from rknnlite.api import RKNNLite

    # Игнорируем SIGINT в дочернем процессе — shutdown управляется через Event
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.sched_setaffinity(0, {4 + core_id})

    try:
        rknn = RKNNLite()
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            print(f"[NPU {core_id}] ❌ Failed to load model", flush=True)
            return

        core_masks = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]
        ret = rknn.init_runtime(core_mask=core_masks[core_id])
        if ret != 0:
            print(f"[NPU {core_id}] ❌ Failed to init runtime", flush=True)
            return

        print(f"🚀 NPU Core {core_id} запущен", flush=True)

        while not shutdown_event.is_set():
            try:
                # ✅ Неблокирующее получение задачи с таймаутом
                task = input_queue.get(timeout=QUEUE_TIMEOUT)
            except Exception:
                continue  # Таймаут или очередь пуста — продолжаем цикл

            if task is None:  # Сигнал завершения
                break

            frame_id, rgb_array = task  # (640, 640, 3), uint8

            # Вход: NHWC (1, 640, 640, 3)
            input_data = np.expand_dims(rgb_array, axis=0)

            # Инференс с замером времени
            start_time = time.perf_counter()
            outputs = rknn.inference(inputs=[input_data])
            inference_time_ms = (time.perf_counter() - start_time) * 1000.0
            out = np.squeeze(outputs[0])

            # Постобработка
            boxes_raw = out[:4, :].transpose()   # (8400, 4)
            raw_scores = out[4:, :].transpose()  # (8400, 80)

            confidences = np.max(raw_scores, axis=1)
            class_ids = np.argmax(raw_scores, axis=1)

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
                    detections = [(float(boxes[i][0]), float(boxes[i][1]),
                                   float(boxes[i][2]), float(boxes[i][3]),
                                   float(conf[i]), int(c_ids[i]))
                                  for i in indices]

            # ✅ Неблокирующая отправка результата с таймаутом
            try:
                output_queue.put((frame_id, detections, inference_time_ms), timeout=QUEUE_TIMEOUT)
            except Exception:
                # Очередь переполнена или закрыта — пропускаем кадр, не блокируем процесс
                pass

    except Exception as e:
        print(f"[NPU {core_id}] ❌ Error: {e}", flush=True)
    finally:
        # Гарантированная очистка ресурсов RKNN
        try:
            rknn.release()
        except:
            pass
        print(f"[NPU {core_id}] 🛑 Завершён", flush=True)


class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        self.get_logger().info('🔧 Initializing Advanced YOLO Detector...')

        # Параметры
        self.declare_parameter('model_path', 'models/yolo26s-rk3588.rknn')
        self.declare_parameter('camera_id', 1)
        self.declare_parameter('obj_thresh', 0.25)
        self.declare_parameter('nms_thresh', 0.45)
        self.declare_parameter('target_object', 'person')
        self.declare_parameter('enable_auto_follow', True)
        self.declare_parameter('max_linear_speed', 0.3)
        self.declare_parameter('turn_speed', 0.5)

        self.model_path = self.get_parameter('model_path').value
        self.camera_id = self.get_parameter('camera_id').value
        self.obj_thresh = self.get_parameter('obj_thresh').value
        self.nms_thresh = self.get_parameter('nms_thresh').value
        self.target_object = self.get_parameter('target_object').value
        self.enable_auto_follow = self.get_parameter('enable_auto_follow').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.turn_speed = self.get_parameter('turn_speed').value

        self.get_logger().info(f'📦 COCO Classes: {len(COCO_CLASSES)} objects')

        # ROS2 publishers/subscribers
        self.detections_pub = self.create_publisher(Detection2DArray, 'detections', 10)
        self.image_pub = self.create_publisher(Image, 'processed_image', 10)
        self.status_pub = self.create_publisher(String, 'detection_status', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.target_sub = self.create_subscription(String, 'set_target', self.target_callback, 10)
        self.conf_sub = self.create_subscription(Float32, 'set_confidence', self.confidence_callback, 10)

        self.cv_bridge = CvBridge()

        # Камера
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        if not self.cap.isOpened():
            self.get_logger().error(f'❌ Failed to open camera {self.camera_id}')
            raise RuntimeError(f'Failed to open camera {self.camera_id}')

        self.get_logger().info(f'📷 Camera initialized: 640x480 @ 30 FPS')

        # Масштабирование координат: модель (640×640) → камера (640×480)
        self.scale_x = 640 / 640.0  # 1.0
        self.scale_y = 480 / 640.0  # 0.75

        # ✅ Event для координации shutdown между процессами
        self.shutdown_event = mp.Event()

        # Очереди с увеличенным размером для буферизации пиков
        self.prep_queue = mp.Queue(maxsize=12)
        self.result_queue = mp.Queue(maxsize=12)

        # Запуск NPU workers
        num_npu_cores = 3
        self.npu_processes = []
        for i in range(num_npu_cores):
            p = mp.Process(target=npu_process,
                           args=(i, self.prep_queue, self.result_queue,
                                 self.model_path, self.obj_thresh, self.nms_thresh,
                                 self.shutdown_event))
            p.start()
            self.npu_processes.append(p)

        self.get_logger().info(f'🧠 Model: {self.model_path} | NPU cores: {num_npu_cores}')

        # Frame buffer
        self.frame_buffer = {}
        self.frame_buffer_lock = threading.Lock()
        self.frame_id_counter = 0
        self.max_buffer_size = 20  # Увеличен буфер для сглаживания пиков

        # Результаты
        self.current_detections = []
        self.current_frame_id = 0
        self.current_inference_time = 0.0
        self.results_lock = threading.Lock()

        # Collector thread с флагом завершения
        self.collector_running = True
        self.collector_thread = threading.Thread(target=self._collect_results, daemon=True)
        self.collector_thread.start()

        # FPS tracking
        self.fps_counter = {'count': 0, 'start': time.time(), 'fps': 0.0}

        # Main timer
        self.timer = self.create_timer(0.033, self.run_detection)

        self.get_logger().info('✅ Advanced YOLO Detector node STARTED')

    def _collect_results(self):
        """Collector thread с неблокирующим get() и проверкой shutdown."""
        while self.collector_running and rclpy.ok():
            try:
                # ✅ Неблокирующее получение с таймаутом
                result = self.result_queue.get(timeout=QUEUE_TIMEOUT)
                if len(result) == 3:
                    frame_id, detections, inference_time_ms = result
                else:
                    frame_id, detections = result[0], result[1]
                    inference_time_ms = 0.0
                with self.results_lock:
                    self.current_detections = detections
                    self.current_frame_id = frame_id
                    self.current_inference_time = inference_time_ms
            except Exception:
                # Таймаут или ошибка — продолжаем цикл
                continue

    def target_callback(self, msg: String):
        self.target_object = msg.data
        self.get_logger().info(f'🎯 Target changed to: {msg.data}')

    def confidence_callback(self, msg: Float32):
        self.obj_thresh = msg.data
        self.get_logger().info(f'📊 Confidence threshold: {msg.data:.2f}')

    def _scale_detections(self, detections):
        """Масштабирует детекции из пространства модели (640×640) в пространство кадра (640×480)."""
        scaled = []
        for (x1, y1, w, h, score, cls_id) in detections:
            x1_s = x1 * self.scale_x
            y1_s = y1 * self.scale_y
            w_s = w * self.scale_x
            h_s = h * self.scale_y
            # Клампинг координат
            x1_s = max(0.0, min(x1_s, 640.0))
            y1_s = max(0.0, min(y1_s, 480.0))
            w_s = max(0.0, min(w_s, 640.0 - x1_s))
            h_s = max(0.0, min(h_s, 480.0 - y1_s))
            scaled.append((x1_s, y1_s, w_s, h_s, score, cls_id))
        return scaled

    def run_detection(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Preprocess
        if frame.shape[0] != IMG_SIZE or frame.shape[1] != IMG_SIZE:
            resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        else:
            resized = frame
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Store frame
        fid = self.frame_id_counter
        self.frame_id_counter += 1
        with self.frame_buffer_lock:
            self.frame_buffer[fid] = frame.copy()
            if len(self.frame_buffer) > self.max_buffer_size:
                oldest = min(self.frame_buffer.keys())
                del self.frame_buffer[oldest]

        # Send to NPU — неблокирующая отправка
        try:
            if not self.prep_queue.full():
                self.prep_queue.put((fid, rgb), timeout=QUEUE_TIMEOUT)
        except Exception:
            # Очередь переполнена — пропускаем кадр, не блокируем главный поток
            pass

        # Get results
        with self.results_lock:
            det_fid = self.current_frame_id
            detections = list(self.current_detections)

        # Масштабируем детекции
        detections = self._scale_detections(detections)

        # Retrieve display frame
        display_frame = None
        with self.frame_buffer_lock:
            if det_fid in self.frame_buffer:
                display_frame = self.frame_buffer[det_fid].copy()
            elif self.frame_buffer:
                display_frame = self.frame_buffer[max(self.frame_buffer.keys())].copy()

        if display_frame is None:
            display_frame = frame

        # Draw detections
        for (x1, y1, w, h, score, cls_id) in detections:
            x1_i, y1_i = int(x1), int(y1)
            x2_i, y2_i = int(x1 + w), int(y1 + h)
            label = f"{COCO_CLASSES.get(cls_id, cls_id)} {score:.2f}"
            cv2.rectangle(display_frame, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 1)
            cv2.putText(display_frame, label, (x1_i, y1_i - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # FPS
        self.fps_counter['count'] += 1
        elapsed = time.time() - self.fps_counter['start']
        if elapsed > 1.0:
            self.fps_counter['fps'] = self.fps_counter['count'] / elapsed
            self.fps_counter['count'] = 0
            self.fps_counter['start'] = time.time()

        self.publish_results(display_frame, detections)

    def get_zone(self, detections):
        for (x1, y1, w, h, score, cls_id) in detections:
            if COCO_CLASSES.get(cls_id, '') == self.target_object:
                center_x = x1 + w / 2
                normalized = center_x / 640.0
                if normalized < 0.35:
                    return 'LEFT'
                elif normalized > 0.65:
                    return 'RIGHT'
                else:
                    return 'CENTER'
        return 'NONE'

    def publish_results(self, frame, detections):
        if self.image_pub.get_subscription_count() > 0:
            img_msg = self.cv_bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = 'camera'
            self.image_pub.publish(img_msg)

        if self.detections_pub.get_subscription_count() > 0:
            det_array = Detection2DArray()
            det_array.header.stamp = self.get_clock().now().to_msg()
            det_array.header.frame_id = 'camera'

            for (x1, y1, w, h, score, cls_id) in detections:
                det = Detection2D()
                det.header = det_array.header
                det.bbox.center.position.x = float(x1 + w / 2)
                det.bbox.center.position.y = float(y1 + h / 2)
                det.bbox.size_x = float(w)
                det.bbox.size_y = float(h)
                hypothesis = ObjectHypothesis()
                hypothesis.class_id = COCO_CLASSES.get(cls_id, str(cls_id))
                hypothesis.score = score
                hyp_with_pose = ObjectHypothesisWithPose()
                hyp_with_pose.hypothesis = hypothesis
                det.results.append(hyp_with_pose)
                det_array.detections.append(det)

            self.detections_pub.publish(det_array)

        if self.status_pub.get_subscription_count() > 0:
            inference_time = self.current_inference_time
            # Проверяем только целевой объект
            target_found = any(COCO_CLASSES.get(cls_id, '') == self.target_object 
                           for (_, _, _, _, _, cls_id) in detections)
            status = String()
            status.data = json.dumps({
                'target': self.target_object,
                'found': target_found,
                'zone': self.get_zone(detections),
                'count': len(detections),
                'classes': ', '.join([COCO_CLASSES.get(cls_id, str(cls_id)) for _, _, _, _, _, cls_id in detections]),
                'fps': self.fps_counter['fps'],
                'inference_time': inference_time
            })
            self.status_pub.publish(status)

        if self.enable_auto_follow:
            self.publish_cmd_vel(detections, frame.shape[1])

    def publish_cmd_vel(self, detections, frame_width):
        twist = Twist()
        target_found = False

        for (x1, y1, w, h, score, cls_id) in detections:
            if COCO_CLASSES.get(cls_id, '') == self.target_object:
                target_found = True
                center_x = x1 + w / 2
                normalized_center = (center_x / frame_width) - 0.5
                turn = normalized_center * self.turn_speed

                if abs(normalized_center) < 0.08:
                    twist.angular.z = 0.0
                    twist.linear.x = self.max_linear_speed
                else:
                    twist.angular.z = -turn
                    twist.linear.x = self.max_linear_speed * 0.5
                break

        if not target_found:
            twist.angular.z = self.turn_speed * 0.5
            twist.linear.x = 0.0

        self.cmd_vel_pub.publish(twist)

    def _shutdown_npu_workers(self):
        """Корректное завершение NPU-процессов."""
        self.get_logger().info('🛑 Stopping NPU workers...')
        
        # 1. Сигнализируем работникам о завершении
        self.shutdown_event.set()
        
        # 2. Отправляем сигнал завершения в очереди (на случай, если работники ждут задачи)
        for _ in self.npu_processes:
            try:
                self.prep_queue.put_nowait(None)
            except:
                pass
        
        # 3. Ждём завершения процессов с таймаутом
        for i, p in enumerate(self.npu_processes):
            if p.is_alive():
                p.join(timeout=2.0)
                if p.is_alive():
                    self.get_logger().warn(f'NPU process {i} не завершился, отправляем terminate...')
                    p.terminate()
                    p.join(timeout=1.0)
        
        # 4. Очищаем очереди
        while not self.prep_queue.empty():
            try:
                self.prep_queue.get_nowait()
            except:
                break
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except:
                break

    def destroy_node(self):
        self.get_logger().info('🛑 Shutting down YOLO Detector Node...')
        
        # Останавливаем collector thread
        self.collector_running = False
        
        # Корректно завершаем NPU-процессы
        self._shutdown_npu_workers()
        
        # Освобождаем камеру
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        # Закрываем очереди
        try:
            self.prep_queue.close()
            self.prep_queue.join_thread()
            self.result_queue.close()
            self.result_queue.join_thread()
        except:
            pass
        
        super().destroy_node()


def main(args=None):
    # ✅ 'spawn' обязателен для RKNN на Linux
    mp.set_start_method('spawn', force=True)
    
    rclpy.init(args=args)
    node = YoloDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
