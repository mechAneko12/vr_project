from django.contrib.postgres.fields import ArrayField
from django.db import models

# Create your models here.

class Wavelet(models.Model):
    wavelet = ArrayField(models.FloatField(), size=240)
    class_label = models.IntegerField()

class Weight(models.Model):
    inpt_size = models.IntegerField(null=True)
    hidden_size = models.IntegerField(null=True)
    output_size = models.IntegerField(null=True)
    layer_num = models.IntegerField(null=True)
    affine1 = ArrayField(ArrayField(models.FloatField()))
    bias1 = ArrayField(models.FloatField(),null=True)
    affine2 = ArrayField(ArrayField(models.FloatField()))
    bias2 = ArrayField(models.FloatField(),null=True)
    #affine1 = ArrayField(ArrayField(models.FloatField(), size=300), size=240)
    #bias1 = ArrayField(models.FloatField(), size=300)
    #affine2 = ArrayField(ArrayField(models.FloatField(), size=300), size=6)
    #bias2 = ArrayField(models.FloatField(), size=6)