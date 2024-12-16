from django.urls import path

from . import views
from .views import ImageUploadView, RatingSaveView

urlpatterns = [
    path('', views.home, name='home'),
    path('home/', views.home, name='home'),
    path('sign-up/', views.sign_up, name='sign_up'),
    path('results/', views.results, name='results'),
    path('save-rating/', RatingSaveView.as_view(), name='save_rating'),
    path('logout/', views.logout_user, name="logout"),
    path('api/process-image/', ImageUploadView.as_view(), name='defects'),
    path('download-report/', views.generate_pdf, name='download_report'),
    path('metrics/', views.metrics, name='metrics')
]