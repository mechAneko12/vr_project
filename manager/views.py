from django.shortcuts import render
from django.http.response import JsonResponse
from django.http import HttpResponse
from .models import Wavelet, Weight
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
json_open = open('staticfiles/jsonfiles/weight_data.json', 'r')
weights = json.load(json_open)
input_size = int(weights['input_size'])
hidden_size = int(weights['hidden_size'])
output_size = int(weights['output_size'])

layers = OrderedDict()
for lay_name in weights['layer_name']:
    if lay_name[0] == "D":
        layers[lay_name] = Dropout(dropout_ratio=0.5)
    elif lay_name[0] == "A":
        layers[lay_name] = Affine(weight=np.array(weights[lay_name]), bias=np.array(weights[lay_name+'_bias']))
    elif lay_name[0] == "B":
        layers[lay_name] = BatchNormalization()
    elif lay_name[0] == "R":
        layers[lay_name] = Relu()

N = weights['N']
def return_class(request):
    global layers, N
    emg_arr_raw = request.POST.getlist('emg_arr[]')
    
    emg_arr = np.empty((0, 8))
    for row_raw in emg_arr_raw:
        row_raw.split(",")
        row = np.array([[int(x) for x in row_raw.split(",")]])
        emg_arr = np.concatenate([emg_arr, row], axis=0)

    dwt = DWT_N(N)
    dwt.filter()
    wavelet = np.empty((0, (N-2)*8))
    for i in range(emg_arr.shape[0] - N):
        f = copy.copy(emg_arr[i:i+N][:])
        dwt.main(f)
        wavelet = np.concatenate([wavelet, f[:][2:].T.reshape(1,(N-2)*8)], axis=0)
    predicted_data = predict(layers, wavelet, train_flg=False)
    predicted_class = predicted_data.argmax(axis=1)
    
    d = {"predicted_class": str(predicted_class)}
    return JsonResponse(d)

def predict(layers, x, train_flg=True):
    for layer in layers.values():
        if isinstance(layer, Dropout) or isinstance(layer, BatchNormalization):
            x = layer.forward(x, train_flg)
        else:
            x = layer.forward(x)
    return x
