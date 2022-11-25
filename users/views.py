import email
import re
from django.shortcuts import render
from .models import User
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import redirect
# Create your views here.

def Glauben(request):
    
    if  request.method == "POST":
        print(request.POST) 
        username = request.POST["username"]
        email = request.POST["email"]
        password = request.POST["password"]
        User.objects.create_user(username, email, password)  
    return render(request, "users/Glauben.html")

def login_view(request):

    return render(request, "users/login.html")

def login_Sidebar(request):

    return render(request, "users/sidebarGlauben.html")

def graficoBar(request):

    return render(request, "users/estadistica.html")

def GlaubenLogin_view(request):
    
    if request.method == "POST":
        email = request.POST["email"]
        password = request.POST["password"]
        user = authenticate(username=email, password=password)
        if user is not None:
            login(request, user)
            print("auteticacion exito")
            return render(request, "users/sidebarGlauben.html")
        else:
            print("auteticacion fallado")            
    return render(request, "users/GlaubenLogin.html")