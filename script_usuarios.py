from django.contrib.auth.models import User
import csv


with open('usuarios.csv', newline='') as f:
    reader = csv.reader(f, delimiter=',')
    usuarios = list(reader)

for x in usuarios:
    user = User.objects.create_user(x[1], 'user@andes.com', 'password')
    # Update fields and then save again
    user.first_name = x[1]
    user.last_name = x[0]
    user.save()
    print(x[1])


# Create user and save to the database
