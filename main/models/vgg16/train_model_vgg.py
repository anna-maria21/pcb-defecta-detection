from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import classification_report

# Переконайтеся, що Pillow встановлено
try:
    from PIL import Image
except ImportError:
    raise ImportError("Pillow не встановлено. Виконайте 'pip install Pillow' для установки.")

# Завантаження передтренованої моделі VGG16 без останніх шарів
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Заморожуємо шари базової моделі, щоб не тренувати їх
for layer in base_model.layers:
    layer.trainable = False

# Додаємо власні шари
x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(6, activation='softmax')(x)  # Припустимо, що у вас 3 типи дефектів

# Створюємо кінцеву модель
model = Model(inputs=base_model.input, outputs=predictions)

# Компіляція моделі
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Створення генератора зображень для аугментації
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

# Завантаження даних для тренування
train_generator = train_datagen.flow_from_directory('D:\magister_work\pcb-defects-detection\pcb_defects_detection\cutting-dataset-images\cropped_images', target_size=(224, 224), batch_size=32, class_mode='categorical')
steps_per_epoch = train_generator.samples
# Навчання моделі
model.fit(train_generator, epochs=20, steps_per_epoch=steps_per_epoch)
model.save('./models/vgg_model.keras')
test_datagen = ImageDataGenerator(rescale=1./255)  # Нормалізація зображень
