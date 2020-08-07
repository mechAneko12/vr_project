from django.contrib.postgres.fields import ArrayField
from django.db import models

# Create your models here.

class Wavelet(models.Model):
    wavelet = ArrayField(models.FloatField(), size=240)
    class_label = models.IntegerField()

class Weights(models.Model):
    affine1 = ArrayField(ArrayField(models.FloatField(), size=1), size=1)
    affine2 = ArrayField(ArrayField(models.FloatField(), size=300), size=6)