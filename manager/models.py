from django.contrib.postgres.fields import ArrayField
from django.db import models

# Create your models here.

class Data(models.Model):
    wavelet = ArrayField(ArrayField(models.FloatField()))
    class_label = models.IntegerField()