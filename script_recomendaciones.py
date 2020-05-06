from Recomendations.models import Recommendation
import csv


with open('rec_finales.csv', newline='', encoding = 'unicode_escape') as f:
    reader = csv.reader(f, delimiter=',')
    recom = list(reader)
#print(opinion)
y=0
for x in recom:
    print(y,"\n")
    #print(float(x[4]),"\n")
    recommendation = Recommendation(user_id=x[0],bussiness_id=x[1],
                                    stars=float(x[5]),
                                    bussines_name=x[2],city=x[4],
                                    address=x[3], review_count=x[6])
    recommendation.save()
    y=y+1
