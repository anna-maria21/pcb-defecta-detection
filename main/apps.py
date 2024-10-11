from django.apps import AppConfig
import main.utils as utils

class MainConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'main'

    def ready(self):
        yolo_model = utils.load_yolo_model()
        faster_r_cnn_model = utils.load_faster_r_cnn_model()
        vgg16_model = ''
        self.models_dict = {'1': yolo_model, '2': faster_r_cnn_model, '3': vgg16_model}

        print("Server has started, and the variable is initialized.")
