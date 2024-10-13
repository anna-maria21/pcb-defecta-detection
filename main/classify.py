from tensorflow.keras.preprocessing import image
import numpy as np


def classify(image_path, model):
    img = image.load_img(image_path, target_size=(60, 60))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    return predicted_class
