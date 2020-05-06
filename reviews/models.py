from django.db import models

# Create your models here.
class reviews(models.Model):
    user_id = models.CharField(max_length=50)
    name = models.CharField(max_length=10)
    bussiness_name = models.CharField(max_length=20)
    city = models.CharField(max_length=50)
    text = models.TextField(max_length=500, default='')
    stars = models.DecimalField(max_digits=8, decimal_places=2, default=0.0)
    date = models.CharField(max_length=50)

    def __str__(self):
        return self.name
