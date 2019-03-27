from django.db import models


class Stats(models.Model):
    spots_empty = models.IntegerField(default=0)
    spots_occupied = models.IntegerField(default=0)
    spots_total = models.IntegerField(default=0)
