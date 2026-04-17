import kagglehub
import os

# Скачиваем датасет
path = kagglehub.dataset_download("adilshamim8/face-detection")
print("Путь к датасету:", path)

# Для удобства выведем содержимое скачанной папки
print("\nСодержимое папки датасета:")
for root, dirs, files in os.walk(path):
    level = root.replace(path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # покажем не более 5 файлов в папке
        print(f"{subindent}{file}")
    if len(files) > 5:
        print(f"{subindent}... и ещё {len(files)-5} файлов")