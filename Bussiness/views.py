from django.shortcuts import render

from django.views.generic.list import ListView

from .models import Bussiness
from Recomendations.models import Recommendation

# Create your views here.

class BussinessSearchListView(ListView):
    template_name = 'Bussiness/search.html'
    paginate_by = 10
    def get_queryset(self):
        #print(Bussiness.objects.filter(city__icontains=self.query()))

        if self.request.user.is_anonymous:
            return Bussiness.objects.filter(city__icontains=self.query()).order_by('-stars')
        else:
            Usuario=self.request.user
            print(Usuario)
            return Recommendation.objects.filter(city__icontains=self.query()).filter(user_name__icontains=Usuario)

    def query(self):
        return self.request.GET.get('q')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['query'] = self.query()
        print(context)
        return context
