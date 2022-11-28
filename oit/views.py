import firebase_admin
from django.http import HttpResponse
from django.shortcuts import render
import pyrebase
from firebase_admin import db, credentials
from .analysis import pressure_map, get_players, get_teams, ball_receipt_map, shot_map, heatmap2, team_pressure, \
    team_shots, xg, pass1, pass2, convex_hull, carry_map

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
cred = credentials.Certificate('oit-project-e7f75-firebase-adminsdk-s34rb-c5e70865cd.json')

# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://oit-project-e7f75-default-rtdb.firebaseio.com'
})

firebase = pyrebase.initialize_app(config)
database = firebase.database()
authe = firebase.auth()

def games(request):
    game_name = database.child('Data').child('Name').get().val()
    ref = db.reference("/Games/")
    games = ref.get()
    print(games)
    list = games.items()

    context = {
        'games': games,
    }
    return render(request, 'oit_app/games.html', context)


def home(request):
    ref = db.reference("/Players/")
    players = ref.get()

    context = {'players': players}

    return render(request, 'oit_app/squad.html', context)


def post(request):
    # players = Customer.objects.all().order_by('number')
    # posts = Post.objects.all().order_by('-created_on')

    ref = db.reference("/Notes/")
    ref1 = db.reference("/Players/")

    notes = ref.get()
    players = ref1.get()
    print(notes)
    return render(request, 'oit_app/notes.html', {'notes': notes, 'players': players})

def player(request, player_id):

    ref = db.reference("/Players/")
    players = ref.get()
    player=""

    for key, value in players.items():
        if value['Id'] == int(player_id):
            player = value

    context = {'player': player}
    return render(request, 'oit_app/player.html', context)


def full_post(request, slug):
    # post = Post.objects.get(id=slug)

    ref = db.reference("/Notes/")
    notes = ref.get()
    post = ""

    for key, value in notes.items():
        if value['Slug'] == slug:
            post = value

    print(post)
    context = {'post': post}
    return render(request, 'oit_app/post_detail.html', context)


def game_analysis(request, game_code):
    teams = get_teams(game_code)

    chart1 = ''
    chart2 = ''
    chart3 = ''
    chart4 = ''
    chart5 = ''
    chart6 = ''

    half1 = 'Первый тайм'
    half2 = 'Второй тайм'
    full_game = '90 минут'

    players1 = get_players(game_code, teams[0])
    players2 = get_players(game_code, teams[1])

    if request.method == 'POST':
        player1 = request.POST.get('player_team1')
        player2 = request.POST.get('player_team2')
        team_stats = request.POST.get('stats')
        first_half = request.POST.get('first_half')
        second_half = request.POST.get('second half')
        both_halves = request.POST.get('90min')

        if player1 is not None:
            chart1 = pass1(game_code, player1, teams[0], 'home')
            chart2 = convex_hull(game_code, player1, 'home')
            chart3 = carry_map(game_code, player1, 'home')
            chart4 = shot_map(game_code, player1, teams[0], 'home')
            chart5 = pressure_map(game_code, player1, teams[0])
            chart6 = ball_receipt_map(game_code, player1, teams[0])
        elif player2 is not None:
            chart1 = pass1(game_code, player2, teams[1], 'away')
            chart2 = convex_hull(game_code, player2, 'away')
            chart3 = carry_map(game_code, player2, 'away')
            chart4 = shot_map(game_code, player2, teams[1], 'away')
            chart5 = pressure_map(game_code, player2, teams[1])
            chart6 = ball_receipt_map(game_code, player2, teams[1])
        elif team_stats is not None:
            # chart1 = xg(game_code)
            chart2 = heatmap2(game_code)
        elif first_half is not None:
            chart1 = team_shots(game_code, teams[0], 'red', half1)
            chart2 = team_shots(game_code, teams[1], 'blue', half1)
            chart3 = team_pressure(game_code, teams[0], 'red', half1)
            chart4 = team_pressure(game_code, teams[1], 'blue', half1)
        elif second_half is not None:
            chart1 = team_shots(game_code, teams[0], 'red', half2)
            chart2 = team_shots(game_code, teams[1], 'blue', half2)
            chart3 = team_pressure(game_code, teams[0], 'red', half2)
            chart4 = team_pressure(game_code, teams[1], 'blue', half2)
        elif both_halves is not None:
            chart1 = team_shots(game_code, teams[0], 'red', full_game)
            chart2 = team_shots(game_code, teams[1], 'blue', full_game)
            chart3 = team_pressure(game_code, teams[0], 'red', full_game)
            chart4 = team_pressure(game_code, teams[1], 'blue', full_game)
            chart5 = pass2(game_code, teams[0], teams[1], 'red')
            chart6 = pass2(game_code, teams[1], teams[0], 'purple')

    context = {'chart1': chart1, 'chart2': chart2, 'chart3': chart3, 'chart4': chart4, 'chart5': chart5,
               'chart6': chart6,
               'teams': teams, 'players1': players1, 'players2': players2}

    return render(request, 'oit_app/game_analysis.html', context)
