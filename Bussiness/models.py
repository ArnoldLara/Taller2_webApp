from django.db import models

# Create your models here.
# Create your models here.
class Bussiness(models.Model):
    id = models.CharField(max_length=50, primary_key=True)
    name = models.CharField(max_length=50)
    city = models.CharField(max_length=20)
    stars = models.DecimalField(max_digits=8, decimal_places=2, default=0.0)
    categories = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
