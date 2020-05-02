from django.urls import path

from . import views

urlpatterns = [
    path('user',views.ReviewsSearchListView.as_view(),name='review'),
]
