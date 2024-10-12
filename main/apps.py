from django.apps import AppConfig

class MainConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'main'

    def ready(self):
        from pcb_defects_detection.clear_data import handle
        handle()

        import main.utils as utils
        yolo_model = utils.load_yolo_model()
        faster_r_cnn_model = utils.load_faster_r_cnn_model()
        vgg16_model = ''
        self.models_dict = {'1': yolo_model, '2': faster_r_cnn_model, '3': vgg16_model}

