import base64
import hashlib
import logging
import os
import cv2

from django.contrib.auth import login, logout
from django.shortcuts import redirect
from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from main.forms import RegistrationForm
from main.localize import localize
from main.models import PcbImage, Location, Defect
from main.utils import draw_bboxes


def home(request):
    return render(request, 'index.html')


def sign_up(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            request.session['current_user_id'] = user.id
            login(request, user)
            return redirect('/home')
    else:
        form = RegistrationForm()

    return render(request, 'registration/sign_up.html', {"form": form})


class ImageUploadView(APIView):
    def post(self, request, *args, **kwargs):
        image_data = request.data.get('image')
        localization_model = request.data.get('localization_model')
        classification_model = request.data.get('classification_model')

        header, encoded = image_data.split(',', 1)
        image_format = header.split('/')[1].split(';')[0]
        image_bytes = base64.b64decode(encoded)

        image_hash = hashlib.sha256(image_bytes).hexdigest()
        pcb_image = PcbImage.objects.filter(image_hash=image_hash).first()
        defects = None
        try:
            defects = Defect.objects.filter(
                image_id=pcb_image.id,
                classification_model_id=int(classification_model),
                localization_model_id=int(localization_model)
            )
        except Exception as ignored:
            pass

        try:
            if pcb_image is None or defects is None:
                image_path = 'D:/магістерська/pcb_defects_detection/main/uploaded_images/'
                pcb_image = PcbImage.objects.create(
                    photo_location=image_path,
                    image_hash=image_hash
                )
                image_name = f"pcb_image_{pcb_image.id}.{image_format}"
                output_path = os.path.join(image_path, image_name)
                with open(output_path, "wb") as image_file:
                    image_file.write(image_bytes)

                message="New image received, processed"
                user_id = request.session.get('current_user_id')

                result = localize(image_path + image_name, localization_model, classification_model, pcb_image, user_id)
            else:
                locations = Location.objects.filter(image_id=pcb_image.id).values_list('x_min', 'y_min', 'x_max', 'y_max')

                classes = Defect.objects.filter(
                    image_id=pcb_image.id,
                    classification_model_id=int(classification_model),
                    localization_model_id=int(localization_model)
                ).values_list('type_id', flat=True)

                image_name = f"pcb_image_{pcb_image.id}.{image_format}"
                image_path = 'D:/магістерська/pcb_defects_detection/main/uploaded_images/'
                output_path = os.path.join(image_path, image_name)
                img = cv2.imread(output_path)
                image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                message="Already processed image, results retrieved from the database"
                result = draw_bboxes(image_rgb, locations, classes)
            return Response({
                             "message": message,
                             "image_id": pcb_image.id,
                             "result": result,
                             "classification_model": classification_model,
                             "localization_model": localization_model
                             },
                status=status.HTTP_201_CREATED
            )
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def classify(image):
    classify(image)


def process(request):
    logging.debug(request)


def logoutUser(request):
    logout(request)
    return redirect('login')
