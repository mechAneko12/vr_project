from django.contrib.postgres.fields import ArrayField
from django.db import models

# Create your models here.

class Wavelet(models.Model):
    wavelet = ArrayField(models.FloatField(), size=240)
    class_label = models.IntegerField()

class Weight(models.Model):
    affine1 = ArrayField(ArrayField(models.FloatField(), size=5), size=5)
    affine2 = ArrayField(ArrayField(models.FloatField(), size=5), size=5)