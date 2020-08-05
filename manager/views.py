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
def index(request):
  template_name = "index.html" # templates以下のパスを書く
  return render(request,template_name)