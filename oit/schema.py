import graphene
from graphene_django import DjangoObjectType
from .models import Book, Player


class BooksType(DjangoObjectType):
    class Meta:
        model = Book
        fields = ("id", "title", "content", "author")


class PlayersType(DjangoObjectType):
    class Meta:
        model = Player
        fields = ("id", "name", "picture", "age", "birthday")


class Query(graphene.ObjectType):
    all_books = graphene.List(BooksType)
    book_by_author = graphene.List(BooksType, id=graphene.Int())

    def resolve_all_books(root, info):
        return Book.objects.all()

    def resolve_book_by_author(self, info, id):
        player_obj = Player.objects.get(id=id)
        return Book.objects.filter(author=player_obj)


class BookMutation(graphene.Mutation):
    class Arguments:
        title = graphene.String(required=True)
        content = graphene.String(required=True)
        author = graphene.Int()

    book = graphene.Field(BooksType)

    @classmethod
    def mutate(cls, root, info, title, content, author):
        player_obj = Player.objects.get(id=author)
        book = Book(title=title, content=content, author=player_obj)
        book.save()
        return BookMutation(book=book)


class BookUpdateMutation(graphene.Mutation):
    class Arguments:
        title = graphene.String(required=True)
        content = graphene.String(required=True)

    book = graphene.Field(BooksType)

    @classmethod
    def mutate(cls, root, info, title, content):
        book = Book.objects.get(title=title)
        book.content = content
        book.save()
        return BookMutation(book=book)


class BookDeleteMutation(graphene.Mutation):
    class Arguments:
        title = graphene.String(required=True)

    book = graphene.Field(BooksType)

    @classmethod
    def mutate(cls, root, info, title):
        book = Book.objects.get(title=title)
        book.delete()
        return


class PlayerMutation(graphene.Mutation):
    class Arguments:
        name = graphene.String(required=True)
        picture = graphene.String(required=True)
        age = graphene.String(required=True)
        birthday = graphene.String(required=True)

    player = graphene.Field(PlayersType)

    @classmethod
    def mutate(cls, root, info, name, picture, age, birthday):
        player = Player(name=name, picture=picture, age=age, birthday=birthday)
        player.save()
        return PlayerMutation(player=player)


class PlayerUpdateMutation(graphene.Mutation):
    class Arguments:
        name = graphene.String(required=True)
        picture = graphene.String(required=True)
        age = graphene.String(required=True)

    player = graphene.Field(PlayersType)

    @classmethod
    def mutate(cls, root, info, name, picture, age):
        player = Player.objects.get(name=name)
        player.picture = picture
        player.age = age
        player.save()
        return PlayerMutation(player=player)


class PlayerDeleteMutation(graphene.Mutation):
    class Arguments:
        name = graphene.String(required=True)

    player = graphene.Field(PlayersType)

    @classmethod
    def mutate(cls, root, info, name):
        player = Player.objects.get(name=name)
        player.delete()
        return


class Mutation(graphene.ObjectType):
    saveBook = BookMutation.Field()
    updateBook = BookUpdateMutation.Field()
    deleteBook = BookDeleteMutation.Field()
    savePlayer = PlayerMutation.Field()
    updatePlayer = PlayerUpdateMutation.Field()
    deletePlayer = PlayerDeleteMutation.Field()


schema = graphene.Schema(query=Query, mutation=Mutation)
