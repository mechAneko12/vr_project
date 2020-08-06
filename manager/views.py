from django.shortcuts import render
from django.http.response import JsonResponse
from django.http import HttpResponse
from .models import Data
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
  predicted_class = request.POST.get('emg_arr[]')
  d = {"predicted_class[]": predicted_class}
  return JsonResponse(d)