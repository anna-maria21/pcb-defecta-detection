import os
from PIL import Image

# Шлях до зображень та текстових файлів
image_folder = 'pcb-defect-dataset/train/images/'
text_folder = 'pcb-defect-dataset/train/labels/'
output_folder = 'cropped_images/'
names = ['mouse_bite', 'spur', 'missing_hole', 'short', 'open_circuit', 'spurious_copper']

# Створюємо базову папку для обрізаних зображень, якщо вона не існує
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Функція для обрізки зображення за YOLO-анотацією
def crop_image_yolo(image_path, yolo_params, output_folder, image_name):
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Розбираємо параметри
    class_id, x_center_norm, y_center_norm, width_norm, height_norm = yolo_params

    # Перетворюємо нормалізовані координати на абсолютні
    x_center = int(x_center_norm * img_width)
    y_center = int(y_center_norm * img_height)
    width = int(width_norm * img_width)
    height = int(height_norm * img_height)

    # Обчислюємо координати для обрізки
    left = int(x_center - width / 2)
    upper = int(y_center - height / 2)
    right = int(x_center + width / 2)
    lower = int(y_center + height / 2)

    # Обрізаємо зображення
    cropped_img = img.crop((left, upper, right, lower))

    # Створюємо папку для поточного класу, якщо вона не існує
    class_folder = os.path.join(output_folder, f'{names[int(class_id)]}')
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

    # Формуємо шлях для збереження обрізаного зображення
    output_image_path = os.path.join(class_folder, image_name)

    # Зберігаємо обрізане зображення
    cropped_img.save(output_image_path)
    print(f"Зображення збережено в {output_image_path}")


# Проходимо через всі зображення та відповідні текстові файли
for image_name in os.listdir(image_folder):
    if image_name.endswith(('.png', '.jpg', '.jpeg')):  # Фільтруємо зображення
        image_path = os.path.join(image_folder, image_name)

        # Знаходимо відповідний текстовий файл
        text_file_name = os.path.splitext(image_name)[0] + '.txt'
        text_file_path = os.path.join(text_folder, text_file_name)

        if os.path.exists(text_file_path):
            with open(text_file_path, 'r') as f:
                lines = f.readlines()
                # Парсимо значення з текстового файлу для YOLO
                class_id, x_center, y_center, width, height = map(float, lines[0].split())

            # Обрізаємо зображення на основі YOLO-анотацій та розміщуємо у відповідній папці
            crop_image_yolo(image_path, (class_id, x_center, y_center, width, height), output_folder, image_name)
        else:
            print(f"Не знайдено текстового файлу для {image_name}")
