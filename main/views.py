import base64
import hashlib
import logging
import os

from django.contrib.auth import login, logout
from django.shortcuts import redirect
from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from main.forms import RegistrationForm
from main.localize import localize
from main.models import PcbImage


def home(request):
    return render(request, 'index.html')


def sign_up(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('/home')
    else:
        form = RegistrationForm()

    return render(request, 'registration/sign_up.html', {"form": form})


class ImageUploadView(APIView):
    def post(self, request, *args, **kwargs):
        try:
            # Extract data from request
            image_data = request.data.get('image')
            localization_model = request.data.get('localization_model')
            classification_model = request.data.get('classification_model')

            # Decode the base64 image data
            header, encoded = image_data.split(',', 1)
            image_format = header.split('/')[1].split(';')[0]  # Extract the image format (e.g., 'jpeg')
            image_bytes = base64.b64decode(encoded)

            # Save the image to the project folder
            image_name = f"pcb_image_{PcbImage.objects.count() + 1}.{image_format}"
            image_path = 'D:/магістерська/pcb_defects_detection/main/uploaded_images/'
            output_path = os.path.join(image_path, image_name)

            # Write the decoded bytes to the file
            with open(output_path, "wb") as image_file:
                image_file.write(image_bytes)

            # Save data to the database
            pcb_image = PcbImage.objects.create(
                photo_location=image_path,
                image_hash=hashlib.sha256(image_bytes).hexdigest()
            )

            localize(image_path + image_name, localization_model, classification_model)

            return Response({"message": "Image uploaded successfully", "image_id": pcb_image.id}, status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def classify(image):
    classify(image)


def process(request):
    logging.debug(request)


def logoutUser(request):
    logout(request)
    return redirect('login')
