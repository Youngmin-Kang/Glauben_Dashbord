import email
import re
from django.shortcuts import render, redirect
from .models import User, Prediccion
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import redirect
from django.template import loader
from django.contrib import messages
from django.contrib.auth.decorators import login_required
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, BatchNormalization,\
                                    LSTM, Dropout, GRU, SimpleRNN,\
                                    InputLayer, Conv1D, MaxPooling1D,\
                                    AveragePooling1D, Flatten
from keras.regularizers import l1, l2, l1_l2
from keras.optimizers import Adam, Adagrad, Adamax, Adadelta, SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt
from dotenv import load_dotenv
import plotly.graph_objects as go
from os.path import join, exists, isfile, isdir
from sklearn.preprocessing import MinMaxScaler
# Create your views here.
@login_required(login_url='login')

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

def prediccion(request):
    
    temp = request.POST['temp']
    conduc = request.POST['conduc']
    difer = request.POST['difer']
    flujoA = request.POST['flujoA']

    df_data = pd.read_csv("./users/data/sdi_training_dataset_4.csv")
    df_sdi = df_data[
        [
            "Temperatura entrada",
            "Conductividad de permeado",
            "Diferencial de Filtros Cartucho",
            "Flujo de Alimentacion",
        ]
    ].values
    X_scaler = MinMaxScaler()
    X_scaler.fit(df_sdi)

    opt = Adam(learning_rate=0.001) # Adagrad, Adadelta, Adamax, Adam, RMSprop, SGD, etc.
    mae  = tf.keras.losses.MeanAbsoluteError()
    rmse = tf.keras.metrics.RootMeanSquaredError()
    mape = tf.keras.losses.MeanAbsolutePercentageError()
    _metrics = [mae, rmse, mape]
    model = tf.keras.Sequential()
    model.add(LSTM(units=256, batch_input_shape=(None, 4, 1), return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(units=64, return_sequences=False)) 
    model.add(Dropout(0.25))
    model.add(Dense(units=1, activation='linear'))
    model.compile(loss='mae', optimizer=opt, metrics=_metrics)
    model.load_weights("./users/models/EXP #29.hdf5")

    model_input = np.hstack((temp, conduc, difer, flujoA))
    model_input = model_input.reshape(-1, 4)
    model_input_norm = X_scaler.transform(model_input)
    model_input_norm = model_input_norm.reshape(1, 4, 1)
    try:
        prediction = model.predict(model_input_norm)
    except Exception as e:
        print(e)
        pass
    prediction = round(prediction[0][0], 2)

    print(prediction)
    context = {
        "temp" : temp,
        "conduc" : conduc,
        "difer" : difer,
        "flujoA" : flujoA,
        "pred" : prediction
    }
    #prediction = Prediccion.objects.create(pred = prediction, temp = temp,conduc = conduc, difer = difer, flujoA = flujoA, user = User.objects.get(username=request.user.username))
    return render(request, "users/sidebarGlauben.html", context)

def register(request):
    list(messages.get_messages(request))
    if request.method == "POST":
        uname = request.POST["username"]
        email = request.POST["email"]
        password = request.POST["password"]
        passwordC = request.POST["passwordC"]
        if password == passwordC:
            user = User.objects.create_user(username = email, email = email, password = password, first_name=uname)
            user.save()
            print(f'Cuenta con correo: {email} creada')
            return redirect('sidebarGlauben')
        else:
            messages.error(request, "Las contraseña debe ser la misma en ambas casillas.")
            print("mensaje añadido")
            return render(request, "users/GlaubenLogin.html")
    else:
        return render(request, "users/GlaubenLogin.html")
    

def glauben_login(request):
    print('Login en proceso')
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        print(email," ",password)
        user = authenticate(request, username=email, password=password)
        print(user)
        if user is not None:
            login(request, user)
            return redirect('sidebarGlauben')
    return redirect("éxito")

def glauben_logout(request):
    logout(request)
    return redirect("login1")