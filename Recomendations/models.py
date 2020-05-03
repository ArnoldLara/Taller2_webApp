from django.db import models

# Create your models here.
class Recommendation(models.Model):
    user_id = models.CharField(max_length=50)
    bussiness_id = models.CharField(max_length=50)
    stars = models.DecimalField(max_digits=8, decimal_places=2, default=0.0)
    city = models.CharField(max_length=20)
    address = models.CharField(max_length=20)
    categories = models.CharField(max_length=20)
