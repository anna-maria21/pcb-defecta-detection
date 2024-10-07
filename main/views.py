import hashlib

from django.core.files.uploadedfile import UploadedFile
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required

from .forms import RegistrationForm
from django.contrib.auth import login, logout, authenticate
# from classify import classify

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import render
import logging

from .models import PcbImage
from .serializers import ProcessedImageSerializer

def home(request):
    return render(request, 'index.html')

def logoutUser(request):
    logout(request)
    return render(request, 'registration/login.html')

@login_required(login_url='/login')
def home_logged(request):
    return render(request, 'index.html')

def sign_up(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('/home-logged')
    else:
        form = RegistrationForm()

    return render(request, 'registration/sign_up.html', {"form": form})


class ImageUploadView(APIView):
    def post(self, request, *args, **kwargs):
        image_file = request.FILES.get('image')

        if not image_file:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Calculate the hash of the image
        image_hash = self.get_image_hash(image_file)

        # Check if the image has been processed before
        if PcbImage.objects.filter(image_hash=image_hash).exists():
            return Response({"message": "Image has already been processed"}, status=status.HTTP_200_OK)

        # Process the image (this is where you'd put your image processing code)
        # Example: save metadata and mark as processed
        processed_image = PcbImage.objects.create(
            photo_name=image_file.name,
            image_hash=image_hash,
            photo_location='D:/магістерська/pcb_defects_detection/main/uploaded_images/'
        )

        serializer = ProcessedImageSerializer(processed_image)
        # classify(image_file)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    def get_image_hash(self, image_file: UploadedFile) -> str:
        hasher = hashlib.sha256()
        for chunk in image_file.chunks():
            hasher.update(chunk)
        return hasher.hexdigest()


def process(request):
    logging.info('jhjh')
    return render(request, 'index.html')