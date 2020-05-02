
from django.contrib import admin
from django.urls import path
from django.urls import include
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.index,name='index'),
    path('usuarios/login',views.login_view,name='login'),
    path('usuarios/logout',views.logout_view,name='logout'),
    path('usuarios/registro',views.registro,name='registro'),
    path('bussiness/', include('Bussiness.urls')),
    path('review/', include('reviews.urls')),
]
