# Generated by Django 2.2.3 on 2020-05-06 02:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Recomendations', '0004_auto_20200506_0237'),
    ]

    operations = [
        migrations.AddField(
            model_name='recommendation',
            name='review_count',
            field=models.DecimalField(decimal_places=2, default=0.0, max_digits=8),
        ),
    ]
