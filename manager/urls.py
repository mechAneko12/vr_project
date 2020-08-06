from django.urls import path
from . import views

urlpatterns = [
  path("",views.index,name="index"),
  path("return_class/", views.return_class, name = "return_class"),
]