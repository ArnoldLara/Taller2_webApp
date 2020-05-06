from django.db import models

# Create your models here.
class Recommendation(models.Model):
    user_id = models.CharField(max_length=50)
    bussiness_id = models.CharField(max_length=50)
    stars = models.DecimalField(max_digits=8, decimal_places=2, default=0.0)
    review_count = models.DecimalField(max_digits=8, decimal_places=0, default=0.0)
    bussines_name = models.CharField(max_length=50, default='')
    city = models.CharField(max_length=50, default='')
    address = models.CharField(max_length=50, default='')

    def __str__(self):
        return self.bussines_name
