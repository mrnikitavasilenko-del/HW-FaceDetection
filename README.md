# HW-FaceDetection

Подсчёт людей, пересекающих заданную линию, с использованием YOLOv8 и DeepSORT.

## 📁 Структура проекта
- `line_crossing_counter.py` – основной скрипт для обработки видео с интерактивным заданием линии.
- `train.py` – обучение модели YOLOv8 на датасете лиц.
- `data.yaml` – конфигурация датасета.
- `convert_csv_to_yolo.py` – конвертация CSV-аннотаций в формат YOLO.
- `download_dataset.py` – загрузка датасета с Kaggle.

## 🚀 Быстрый старт

1. Установите зависимости:
   ```bash
   pip install ultralytics deep-sort-realtime opencv-python
