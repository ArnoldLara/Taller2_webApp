# Generated by Django 2.2.3 on 2020-05-06 17:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('reviews', '0003_auto_20200506_1659'),
    ]

    operations = [
        migrations.AddField(
            model_name='reviews',
            name='text',
            field=models.TextField(default='', max_length=200),
        ),
    ]