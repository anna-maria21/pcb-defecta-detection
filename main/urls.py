
from django.urls import path
from . import views
from .views import ImageUploadView

urlpatterns = [
    path('', views.home, name='home'),
    path('home', views.home, name='home'),
    path('sign-up', views.sign_up, name='sign_up'),
    path('classify', ImageUploadView.as_view(), name='classify'),
]
