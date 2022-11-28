from django.db import models


# Create your models here.

class Player(models.Model):
    name = models.CharField(max_length=100)
    picture = models.CharField(max_length=100)
    age = models.CharField(max_length=100)
    birthday = models.CharField(max_length=100)

    def __str__(self):
        return self.name


class Book(models.Model):
    title = models.CharField(max_length=100)
    content = models.TextField()
    author = models.ForeignKey(Player, on_delete=models.CASCADE, default=1)

    def __str__(self):
        return self.title
