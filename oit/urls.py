from django.urls import path
from graphene_django.views import GraphQLView
from oit.schema import schema
from oit import views

urlpatterns = [
    path('main/', views.games, name='games'),
    path('game_analysis/<str:game_code>', views.game_analysis, name="game_analysis"),
    path('notes/', views.post, name="notes"),
    path('notes/<slug:slug>/', views.full_post, name="post_detail"),
    path('', views.home, name="home"),
    path('player/<str:player_id>/', views.player, name="player"),
    path("graphql/", GraphQLView.as_view(graphiql=True, schema=schema), name="graphql"),
]
