from django.shortcuts import render
from django.http.response import JsonResponse
from django.http import HttpResponse
from .models import Wavelet_Class
import datetime
from django.core import serializers
import json
import numpy as np
import urllib.request

# Create your views here.
#page funnction
def index(request):
  template_name = "index.html" # templates以下のパスを書く
  return render(request,template_name)


#predicted_classを返すfunction

def return_class(request):
  emg_arr_raw = request.POST.getlist('emg_arr[]')
  
  emg_arr = np.empty((0, 8))
  for row_raw in emg_arr_raw:
    row_raw.split(",")
    row = np.array([[int(x) for x in row_raw.split(",")]])
    emg_arr = np.concatenate([emg_arr, row], axis=0)
  
  predicted_class = list(emg_arr)
  d = {"predicted_class": str(predicted_class)}
  return JsonResponse(d)