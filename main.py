#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import multiprocessing as mp
import uvicorn
import time
import os
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse

# --- НАСТРОЙКИ ---
MODEL_PATH = 'yolo26s_rknn_model/yolo26s-rk3588.rknn'
CAMERA_ID = 1
IMG_SIZE = 640
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
NUM_NPU_CORES = 3  # RK3588 имеет ровно 3 NPU ядра

COCO_CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 
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
                78: 'hair drier', 79: 'toothbrush'}

# --- NPU ПРОЦЕСС ---
def npu_process(core_id, input_queue, output_queue):
    """
    Отдельный ПРОЦЕСС для каждого NPU ядра.
    Импорт RKNN делаем здесь, чтобы избежать проблем с fork.
    """
    from rknnlite.api import RKNNLite
    
    # Привязываем процесс к конкретному CPU ядру для стабильности
    os.sched_setaffinity(0, {4 + core_id})  # Используем большие ядра CPU
    
    rknn = RKNNLite()
    rknn.load_rknn(MODEL_PATH)
    
    core_masks = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]
    rknn.init_runtime(core_mask=core_masks[core_id])
    
    print(f"🚀 NPU Core {core_id} запущен")
    
    while True:
        task = input_queue.get()
        if task is None:
            break
            
        frame_id, rgb_frame = task
        
        # Инференс
        outputs = rknn.inference(inputs=[np.expand_dims(rgb_frame, 0)])
        out = np.squeeze(outputs[0])
        
        # Постобработка прямо здесь, чтобы разгрузить главный процесс
        boxes_raw = out[:4, :].transpose()
        probs = out[4:, :].transpose()
        confidences = np.max(probs, axis=1)
        class_ids = np.argmax(probs, axis=1)
        
        mask = confidences > OBJ_THRESH
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
            
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(), OBJ_THRESH, NMS_THRESH)
            if len(indices) > 0:
                indices = indices.flatten()
                detections = [(boxes[i], conf[i], int(c_ids[i])) for i in indices]
        
        output_queue.put((frame_id, rgb_frame, detections))

# --- ЗАХВАТ И ПРЕДОБРАБОТКА ---
def capture_process(prep_queue):
    """Процесс захвата с максимальной скоростью"""
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    print("✅ Камера запущена")
    frame_id = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Предобработка здесь же
        if frame.shape[0] != IMG_SIZE or frame.shape[1] != IMG_SIZE:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if prep_queue.full():
            try:
                prep_queue.get_nowait()
            except:
                pass
        prep_queue.put((frame_id, rgb))
        frame_id += 1

# --- ГЛАВНЫЙ ПРОЦЕСС (СЕРВЕР И РЕНДЕРИНГ) ---
class VideoServer:
    def __init__(self):
        self.app = FastAPI()
        self.frame_buffer = {}  # Буфер для синхронизации результатов
        self.next_frame_id = 0
        self.current_frame = None
        self.lock = mp.Lock()
        
        # Очереди multiprocessing
        self.prep_queue = mp.Queue(maxsize=6)
        self.result_queue = mp.Queue(maxsize=6)
        
        # Запускаем NPU процессы
        self.npu_processes = []
        for i in range(NUM_NPU_CORES):
            p = mp.Process(target=npu_process, args=(i, self.prep_queue, self.result_queue))
            p.start()
            self.npu_processes.append(p)
        
        # Запускаем захват
        self.cap_proc = mp.Process(target=capture_process, args=(self.prep_queue,))
        self.cap_proc.start()
        
        # Поток сбора результатов
        import threading
        threading.Thread(target=self.collector_thread, daemon=True).start()
        
        self.setup_routes()
    
    def collector_thread(self):
        """Собираем результаты от NPU и обновляем текущий кадр"""
        while True:
            frame_id, rgb, detections = self.result_queue.get()
            
            # Рисуем
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for (box, score, cls_id) in detections:
                x1, y1 = int(box[0]), int(box[1])
                x2, y2 = int(box[0] + box[2]), int(box[1] + box[3])
                label = f"{COCO_CLASSES.get(cls_id, cls_id)} {score:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            with self.lock:
                self.current_frame = frame
    
    def setup_routes(self):
        @self.app.get('/video_feed')
        async def video_feed():
            def gen():
                while True:
                    with self.lock:
                        if self.current_frame is None:
                            continue
                        frame = self.current_frame.copy()
                    
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    time.sleep(0.033)  # ~30 FPS для стрима
            return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')

        @self.app.get('/')
        def index():
            return Response(content="""
                <html><body style="margin:0;background:#000;display:flex;justify-content:center;align-items:center;height:100vh;">
                <img src="/video_feed" style="max-width:100%;max-height:100%;">
                </body></html>
            """, media_type="text/html")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Важно для RKNN!
    server = VideoServer()
    print("🚀 Сервер запущен на http://0.0.0.0:8000")
    uvicorn.run(server.app, host="0.0.0.0", port=8000, log_config=None)
