from Recomendations.models import Recommendation
import csv


with open('recommendation.csv', newline='', encoding = 'unicode_escape') as f:
    reader = csv.reader(f, delimiter=';')
    recom = list(reader)
#print(opinion)
y=0
for x in recom:
    print(y,"\n")
    #print(float(x[4]),"\n")
    recommendation = Recommendation(user_id=x[0],bussiness_id=x[1],
                                    stars=float(x[2]),user_name=x[3],
                                    bussines_name=x[4],city=x[5],
                                    address=x[6],categories=x[7])
    recommendation.save()
    y=y+1
