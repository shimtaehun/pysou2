from django.urls import path
from myapp import views

urlpatterns = [
    path('', views.show, name='show'), 
]
# mainapp에 있는 urls.py에 있는 파일을 불러오는 역활