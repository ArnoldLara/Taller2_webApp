#Este archivo nos permite crear formulario a partir de clases
from django import forms

from django.contrib.auth.models import User

#La clase debe de heredar de Forms
class RegisterForm(forms.Form):
    username = forms.CharField(required=True,
                                min_length=4,
                                max_length=50,
                                widget=forms.TextInput(attrs={
                                    'class': 'form-control',
                                    'id': 'username',
                                    'placeholder':'Username'
                                }))
    email = forms.EmailField(required=True,
                                widget=forms.EmailInput(attrs={
                                    'class': 'form-control',
                                    'id': 'email',
                                    'placeholder':'example@codigofacilito.com',
                                }))
    password = forms.CharField(required=True,
                                widget=forms.PasswordInput(attrs={
                                    'class': 'form-control',
                                }))
    password2 = forms.CharField(label='Confirmar password',
                                required=True,
                                widget=forms.PasswordInput(attrs={
                                    'class': 'form-control',
                                }))

    #Importante que la funcion se llame asi para que DJANGO sepa que vamos
    #a hacer una validacion sobre el campo Username
    def clean_username(self):
        username = self.cleaned_data.get('username')

        if User.objects.filter(username=username).exists():
            raise forms.ValidationError('El nombre de usuario ya esta en uso')

        return username

    def clean_email(self):
        email = self.cleaned_data.get('email')

        if User.objects.filter(email=email).exists():
            raise forms.ValidationError('El email ya esta en uso')

        return email

    #Se usa el metodo clean cada vez que necesitemos validar 2 campos que uno dependa
    #del control
    def clean(self):
        cleaned_data = super().clean()

        if cleaned_data.get('password2') !=cleaned_data.get('password'):
            self.add_error('password2','El password no coincide')

    def save(self):
        return User.objects.create_user(
            self.cleaned_data.get('username'),
            self.cleaned_data.get('email'),
            self.cleaned_data.get('password'),
        )
