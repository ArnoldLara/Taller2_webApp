from django.shortcuts import render

from django.views.generic.list import ListView

from .models import Bussiness

# Create your views here.

class BussinessSearchListView(ListView):
    template_name = 'Bussiness/search.html'
    paginate_by = 10
    def get_queryset(self):
        #print(Bussiness.objects.filter(city__icontains=self.query()))
        return Bussiness.objects.filter(city__icontains=self.query()).order_by('-stars')

    def query(self):
        return self.request.GET.get('q')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['query'] = self.query()
        #print(context)
        return context
