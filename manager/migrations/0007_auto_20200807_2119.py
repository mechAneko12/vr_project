# Generated by Django 3.0.8 on 2020-08-07 12:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('manager', '0006_auto_20200807_2112'),
    ]

    operations = [
        migrations.AddField(
            model_name='weight',
            name='hidden_size',
            field=models.IntegerField(null=True),
        ),
        migrations.AddField(
            model_name='weight',
            name='inpt_size',
            field=models.IntegerField(null=True),
        ),
        migrations.AddField(
            model_name='weight',
            name='layer_num',
            field=models.IntegerField(null=True),
        ),
        migrations.AddField(
            model_name='weight',
            name='output_size',
            field=models.IntegerField(null=True),
        ),
    ]
