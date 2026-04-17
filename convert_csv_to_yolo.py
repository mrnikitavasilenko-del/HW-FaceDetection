import os
import pandas as pd

def convert_csv_to_yolo(csv_path, images_dir, labels_dir):
    """
    Читает CSV с колонками: filename, width, height, class, xmin, ymin, xmax, ymax
    Создаёт txt файлы в labels_dir с аннотациями YOLO.
    """
    df = pd.read_csv(csv_path)
    
    # Убедимся, что папка для лейблов существует
    os.makedirs(labels_dir, exist_ok=True)
    
    for _, row in df.iterrows():
        img_filename = row['filename']
        img_width = row['width']
        img_height = row['height']
        class_name = row['class']
        xmin = row['xmin']
        ymin = row['ymin']
        xmax = row['xmax']
        ymax = row['ymax']
        
        # Вычисляем нормализованные параметры YOLO
        box_width = xmax - xmin
        box_height = ymax - ymin
        center_x = xmin + box_width / 2.0
        center_y = ymin + box_height / 2.0
        
        # Нормализуем
        center_x_norm = center_x / img_width
        center_y_norm = center_y / img_height
        width_norm = box_width / img_width
        height_norm = box_height / img_height
        
        # Класс всегда 0 (face)
        class_id = 0
        
        # Имя txt файла без расширения
        base_name = os.path.splitext(img_filename)[0]
        txt_path = os.path.join(labels_dir, base_name + ".txt")
        
        # Дописываем строку в файл
        with open(txt_path, 'a') as f:
            f.write(f"{class_id} {center_x_norm:.6f} {center_y_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")
    
    print(f"Обработан {csv_path}: создано/обновлено аннотаций для {len(df)} объектов")

def process_dataset(base_path):
    """
    Обрабатывает train, valid, test подпапки.
    """
    subsets = ['train', 'valid', 'test']
    for subset in subsets:
        subset_path = os.path.join(base_path, subset, subset)  # из-за структуры train/train
        csv_file = os.path.join(subset_path, '_annotations.csv')
        if os.path.exists(csv_file):
            images_dir = subset_path  # изображения там же
            labels_dir = subset_path  # аннотации кладём рядом
            convert_csv_to_yolo(csv_file, images_dir, labels_dir)
        else:
            print(f"CSV не найден: {csv_file}")

if __name__ == "__main__":
    # Путь к корню датасета
    dataset_root = r"C:\Users\Nikita Vasilenko\.cache\kagglehub\datasets\adilshamim8\face-detection\versions\1"
    process_dataset(dataset_root)