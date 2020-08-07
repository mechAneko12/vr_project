from django.contrib.postgres.fields import ArrayField
from django.db import models

# Create your models here.

class Wavelet_Class(models.Model):
    wavelet = ArrayField(models.FloatField(), size=240)
    class_label = models.IntegerField()

class Weights(models.Model):
    affine1 = ArrayField(ArrayField(models.FloatField()))
    affine2 = ArrayField(ArrayField(models.FloatField()))