import graphene

#from .types import PlayerType
from .models import Player

'''
class EditPlayerMutation(graphene.Mutation):
    class Arguments:
        # The input arguments for this mutation
        id = graphene.ID()
        name = graphene.String()
        picture = graphene.String()
        age = graphene.String()
        birthday = graphene.String()

    # The class attributes define the response of the mutation
   # player = graphene.Field(PlayerType)

    def mutate(self, info, id, name, picture, age, birthday):
        player = Player.objects.get(pk=id)
        player.picture = picture
        player.age = age
        player.birthday = birthday
        player.save()
        # Notice we return an instance of this mutation
        return EditPlayerMutation(player=player)
    '''