from reviews.models import reviews
import csv


with open('reviews.csv', newline='', encoding = 'unicode_escape') as f:
    reader = csv.reader(f, delimiter=';')
    opinion = list(reader)
#print(opinion)
y=0
for x in opinion:
    print(y,"\n")
    #print(float(x[4]),"\n")
    review = reviews(user_id=x[0],name=x[1],bussiness_name=x[2],city=x[3],stars=float(x[4]),date=x[5])
    review.save()
    y=y+1
