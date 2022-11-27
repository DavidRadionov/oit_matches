from django.urls import path

from oit import views

urlpatterns = [
    path('main/', views.games, name='games'),
    path('game_analysis/<str:game_code>', views.game_analysis, name="game_analysis"),
]
