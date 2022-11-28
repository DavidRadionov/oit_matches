import graphene
from graphene_django import DjangoObjectType
from .models import Book


# from .mutations import EditPlayerMutation


class BooksType(DjangoObjectType):
    class Meta:
        model = Book
        fields = ("id", "title", "content")


class Query(graphene.ObjectType):
    all_books = graphene.List(BooksType)

    def resolve_all_books(root, info):
        return Book.objects.all()

    def resolve_book_by_author(self, info, id):
        return Book.objects.get(id=id)


class BookMutation(graphene.Mutation):
    class Arguments:
        title = graphene.String(required=True)

    book = graphene.Field(BooksType)

    @classmethod
    def mutate(cls, root, info, title):
        book = Book(title=title)
        book.save()
        return BookMutation(book=book)


# class Mutation(graphene.ObjectType):
# update_player = graphene.Field(EditPlayerMutation)

class Mutation(graphene.ObjectType):
    update_book = BookMutation.Field()


schema = graphene.Schema(query=Query, mutation=Mutation)
