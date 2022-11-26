import firebase_admin
from django.http import HttpResponse
from django.shortcuts import render
import pyrebase
from firebase_admin import db, credentials

config = {
    "apiKey": "AIzaSyB-f4UqSWgTvJOQwN7_UPEqeJs9hdB7J-Y",
    "authDomain": "oit-project-e7f75.firebaseapp.com",
    "databaseURL": "https://oit-project-e7f75-default-rtdb.firebaseio.com/",
    "projectId": "oit-project-e7f75",
    "storageBucket": "oit-project-e7f75.appspot.com",
    "messagingSenderId": "490547200412",
    "appId": "1:490547200412:web:cbf18b082c639dee130859"
}

# Fetch the service account key JSON file contents
cred = credentials.Certificate('C:/Users/Gleb/Downloads/oit-project-e7f75-firebase-adminsdk-s34rb-c5e70865cd.json')

# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://oit-project-e7f75-default-rtdb.firebaseio.com'
})

firebase = pyrebase.initialize_app(config)
database = firebase.database()
authe = firebase.auth()


def index(request):
    game_name = database.child('Data').child('Name').get().val()
    ref = db.reference("/Games/")
    games = ref.get()
    print(games)
    list = games.items()
    a = []
    for key, value in games.items():
       a += value["Code"]

    context = {
        'channel': game_name, 'a': a
    }
    return render(request, 'oit_app/games.html', context)
