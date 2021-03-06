# Generated by Django 2.2.3 on 2020-04-15 22:52

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Bussiness',
            fields=[
                ('id', models.CharField(max_length=50, primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=50)),
                ('city', models.CharField(max_length=20)),
                ('stars', models.DecimalField(decimal_places=2, default=0.0, max_digits=8)),
                ('categories', models.CharField(max_length=100)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
