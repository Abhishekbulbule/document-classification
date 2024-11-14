# urls.py
from django.urls import path
from .views import DocumentClassificationView

urlpatterns = [
    path('classify/', DocumentClassificationView.as_view(), name='classify'),
]
