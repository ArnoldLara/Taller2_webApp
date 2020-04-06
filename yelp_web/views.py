#from django.http import HttpResponse
from django.shortcuts import render
from .forms import RegisterForm

from django.shortcuts import redirect

from django.contrib.auth import authenticate
from django.contrib.auth import logout
from django.contrib.auth import login
from django.contrib import messages

def index(request):
    return render(request,'index.html')


def login_view(request):

    if request.user.is_authenticated:
        return redirect('index')

    if request.method == 'POST':
        #Los atributos que devuelve el metodo POST los devuelve en un diccionario
        #Asi que se pueden consultar con el metodo get pasando la llave primeria
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(username=username,password=password)#Retorna none si no existe usuario
        if user:
            login(request,user)
            messages.success(request,'Bienvenido {}'.format(user.username))
            #Redirige a la pagina de home
            return redirect('index')

        else:
            messages.success(request,'Usuario o contraseñas no validos')
            print("Usuario no autenticado")

    return render(request,'users/login.html',{

    })

def logout_view(request):
    logout(request)
    messages.success(request,'Sesion cerrada correctamente.')
    return redirect('login')

def registro(request):

    if request.user.is_authenticated:
        return redirect('index')

    form = RegisterForm(request.POST or None)

    if request.method == 'POST' and form.is_valid():
        #Se obtienen los datos con el metodo cleaned data que actua como un diccionario
        user = form.save()
        if user:
            login(request,user)
            messages.success(request,'Usuario creado exitosamente')
            return redirect('index')

    return render(request,'users/register.html',{
        'form':form
    })
