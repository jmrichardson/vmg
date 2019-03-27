from django.contrib import admin
from django.urls import path, include

from pages import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('allauth.urls')),
    path('', views.home, name='home'),
    path('parking_lot_stream', views.parking_lot_stream, name='parking_lot_stream'),
    path('spots_empty_stream', views.spots_empty_stream, name='spots_empty_stream'),
    # path('spots_total_stream', views.spots_total_stream, name='spots_total_stream'),
    path('spots_occupied_stream', views.spots_occupied_stream, name='spots_occupied_stream'),
]
