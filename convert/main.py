import os
from ultralytics import YOLO
from rknn.api import RKNN
from ultralytics.data.utils import check_det_dataset

# --- НАСТРОЙКИ ---
MODEL_NAME = "yolo26s"
PLATFORM = "rk3588"  # Укажи свою: rk3562, rk3566, rk3568, rk3576, rk3588, rv1126
DATASET_YAML = "coco128.yaml"
CALIB_TXT = "dataset_calib_all.txt"
OUTPUT_RKNN = f"{MODEL_NAME}_{PLATFORM}_int8_full.rknn"

def run_full_process():
    # 1. Экспорт YOLOv8s в ONNX (Opset 17)
    print("--> Шаг 1: Экспорт в ONNX (Opset 17)...")
    model = YOLO(f"{MODEL_NAME}.pt")
    onnx_path = model.export(format="onnx", opset=17, simplify=True)
    
    # 2. Подготовка датасета (Все 128 изображений)
    print("--> Шаг 2: Подготовка ВСЕХ изображений coco128 для калибровки...")
    try:
        # Проверяем наличие coco128, если нет - скачается само
        data_info = check_det_dataset(DATASET_YAML)
        img_dir = os.path.join(data_info['path'], 'images', 'train2017')
        
        all_images = [os.path.join(img_dir, img) for img in os.listdir(img_dir) 
                      if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        with open(CALIB_TXT, 'w') as f:
            f.write("\n".join(all_images))
        print(f"Использовано изображений для калибровки: {len(all_images)}")
    except Exception as e:
        print(f"Ошибка подготовки данных: {e}"); return

    # 3. RKNN Конвертация
    print(f"--> Шаг 3: Конвертация в RKNN ({PLATFORM}) с INT8 квантованием...")
    rknn = RKNN(verbose=False)

    # Конфигурация нормализации для YOLOv8
    rknn.config(
        mean_values=[[0, 0, 0]], 
        std_values=[[255, 255, 255]], 
        target_platform=PLATFORM
    )

    if rknn.load_onnx(model=onnx_path) != 0:
        print("Ошибка загрузки ONNX"); return

    # Запуск сборки с использованием ВСЕХ картинок
    # Внимание: с большим количеством фото процесс build может занять несколько минут
    print("Запуск квантования (Full Dataset)...")
    if rknn.build(do_quantization=True, dataset=CALIB_TXT) != 0:
        print("Ошибка сборки RKNN"); return

    if rknn.export_rknn(OUTPUT_RKNN) != 0:
        print("Ошибка экспорта"); return

    print(f"--- ГОТОВО! Файл: {OUTPUT_RKNN} ---")
    rknn.release()

if __name__ == '__main__':
    run_full_process()

