# Generated by Django 4.1.5 on 2023-01-30 21:53

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('manager', '0008_auto_20200807_2143'),
    ]

    operations = [
        migrations.DeleteModel(
            name='Wavelet',
        ),
        migrations.DeleteModel(
            name='Weight',
        ),
    ]
