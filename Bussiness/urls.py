from django.urls import path

from . import views

urlpatterns = [
    path('search',views.BussinessSearchListView.as_view(),name='search'),
]
