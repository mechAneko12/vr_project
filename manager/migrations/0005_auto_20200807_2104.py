# Generated by Django 3.0.8 on 2020-08-07 12:04

import django.contrib.postgres.fields
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('manager', '0004_auto_20200807_2024'),
    ]

    operations = [
        migrations.AlterField(
            model_name='weight',
            name='affine2',
            field=django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=5), size=5),
        ),
    ]