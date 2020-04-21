from Bussiness.models import Bussiness
import csv


with open('bussiness.csv', newline='') as f:
    reader = csv.reader(f, delimiter=',')
    bussiness = list(reader)

y=0
for x in bussiness:
    print(y,"\n")
    #print(float(x[4]),"\n")

    buss=Bussiness(id=x[0],name=x[1],city=x[3],stars=float(x[4]),categories=x[5])
    buss.save()
    y=y+1
