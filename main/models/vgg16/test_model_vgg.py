import kagglehub
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# path = kagglehub.model_download("oleksandrataratonova/vgg16/keras/default")
img_path = '/pcb_defects_detection/cutting-dataset-images/cropped_images/mouse_bite/l_light_01_mouse_bite_15_2_600.jpg'

# Завантажуємо зображення і приводимо його до розміру (224, 224)
img = image.load_img(img_path, target_size=(224, 224))
model = load_model('models\\vgg_model6.keras')

# Перетворюємо зображення в масив і масштабуємо значення пікселів
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Додаємо ось, щоб створити партію з 1 зображення
img_array /= 255.0  # Нормалізація зображення, якщо це потрібно для вашої моделі

# Передбачення класу
predictions = model.predict(img_array)

# Виведення передбаченого класу
predicted_class = np.argmax(predictions, axis=1)
print(f'Передбачений клас: {predicted_class}')