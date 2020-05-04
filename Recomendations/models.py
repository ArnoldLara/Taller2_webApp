from django.db import models

# Create your models here.
class Recommendation(models.Model):
    user_id = models.CharField(max_length=50)
    bussiness_id = models.CharField(max_length=50)
    stars = models.DecimalField(max_digits=8, decimal_places=2, default=0.0)
    user_name = models.CharField(max_length=20, default='')
    bussines_name = models.CharField(max_length=20, default='')
    city = models.CharField(max_length=20, default='')
    address = models.CharField(max_length=20, default='')
    categories = models.CharField(max_length=20, default='')

    def __str__(self):
        return self.bussines_name
