# Generated by Django 2.2.3 on 2020-05-06 02:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Recomendations', '0003_auto_20200503_2343'),
    ]

    operations = [
        migrations.AlterField(
            model_name='recommendation',
            name='address',
            field=models.CharField(default='', max_length=50),
        ),
        migrations.AlterField(
            model_name='recommendation',
            name='bussines_name',
            field=models.CharField(default='', max_length=50),
        ),
        migrations.AlterField(
            model_name='recommendation',
            name='categories',
            field=models.CharField(default='', max_length=150),
        ),
        migrations.AlterField(
            model_name='recommendation',
            name='city',
            field=models.CharField(default='', max_length=50),
        ),
    ]