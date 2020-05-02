from django.shortcuts import render

from django.views.generic.list import ListView

from .models import reviews

from django.contrib.auth.models import User

# Create your views here.

class ReviewsSearchListView(ListView):
    template_name = 'reviews/reviews.html'
    paginate_by = 10
    def get_queryset(self):
        Usuario=self.request.user.last_name
        #print(Usuario)
        #print(Bussiness.objects.filter(city__icontains=self.query()))
        return reviews.objects.filter(user_id__icontains=Usuario)

    def query(self):
        return self.request.GET.get('q')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['query'] = self.query()
        #print(context)
        return context
