from django.apps import AppConfig

class MainConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'main'

    def ready(self):
        # from pcb_defects_detection.clear_data import handle
        # handle()

        self.models_map = {
            1: 'Yolo v8',
            2: 'Faster R-CNN',
            3: 'VGG',
            4: 'ResNet'
        }
        self.color_map = {
            0: (0, 255, 0),  # Green
            1: (255, 0, 0),  # Blue
            2: (0, 0, 255),  # Red
            3: (255, 255, 0),  # Cyan
            4: (255, 0, 255),  # Magenta
            5: (0, 255, 255)  # Yellow
        }
        self.report_colors = {
            0: 'Зелений',
            1: 'Синій',
            2: 'Червоний',
            3: 'Блакитний',
            4: 'Рожевий',
            5: 'Жовтий'
        }
        self.defect_types = {
            0: 'Відсутній отвір (Missing hole)',
            1: 'Мишачий укус (Mouse bite)',
            2: 'Відкритий контур (Open circuit)',
            3: 'Замикання (Short)',
            4: 'Шпора (Spur)',
            5: 'Небажана мідь (Spurious copper)'
        }

        import main.utils as utils
        yolo_model = utils.load_yolo_model()
        faster_r_cnn_model = utils.load_faster_r_cnn_model()
        vgg16_model = utils.load_vgg_model()
        res_net50 = utils.load_resnet_model()
        self.models_dict = {'1': yolo_model, '2': faster_r_cnn_model, '3': vgg16_model, '4': res_net50}

