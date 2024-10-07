from django.urls import path
from . import views
from .views import ImageUploadView

urlpatterns = [
    path('', views.home, name='home'),
    path('home', views.home_logged, name='home_logged'),
    path('sign-up', views.sign_up, name='sign_up'),
    path('classify', ImageUploadView.as_view(), name='classify'),
    path('api/process-image/', views.process),
    path('logout', views.logoutUser, name='logoutUser')
]