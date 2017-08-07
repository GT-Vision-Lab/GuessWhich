# from __future__ import unicode_literals

from django.db import models
from django.core.urlresolvers import reverse
from django.contrib.postgres.fields import ArrayField


class GameRound(models.Model):
    """
    Model depicts the game and each round of the Game
    """
    socket_id = models.CharField(max_length=100, blank=True, null=True)
    worker_id = models.CharField(max_length=100, blank=True, null=True)
    assignment_id = models.CharField(max_length=100, blank=True, null=True)
    level = models.CharField(max_length=100, blank=True, null=True)
    task = models.CharField(max_length=100, blank=True, null=True)
    hit_id = models.CharField(max_length=100, blank=True, null=True)
    game_id = models.CharField(max_length=100, blank=True, null=True)
    round_id = models.CharField(max_length=100, blank=True, null=True)
    question = models.CharField(max_length=100, blank=True, null=True)
    answer = models.CharField(max_length=100, blank=True, null=True)
    history = models.CharField(max_length=10000, blank=True, null=True)
    target_image = models.CharField(max_length=100, blank=True, null=True)
    bot = models.CharField(max_length=100, blank=True, null=True)
    user_picked_image = models.CharField(max_length=100, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.NullBooleanField(default=True, blank=True, null=True)

    def __unicode__(self):
        return "%s : %s : %s" % (self.assignment_id, self.game_id, self.round_id)


class ImageRanking(models.Model):
    socket_id = models.CharField(max_length=100, blank=True, null=True)
    worker_id = models.CharField(max_length=100, blank=True, null=True)
    assignment_id = models.CharField(max_length=100, blank=True, null=True)
    level = models.CharField(max_length=100, blank=True, null=True)
    task = models.CharField(max_length=100, blank=True, null=True)
    hit_id = models.CharField(max_length=100, blank=True, null=True)
    game_id = models.CharField(max_length=100, blank=True, null=True)
    target_image = models.CharField(max_length=100, blank=True, null=True)
    final_image_list = ArrayField(models.CharField(max_length=200), blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    bot = models.CharField(max_length=100, blank=True, null=True)
    score = models.FloatField(default=0)
    is_active = models.NullBooleanField(default=True, blank=True, null=True)

    def __unicode__(self):
        return "%s : %s : %s" % (self.assignment_id, self.game_id, self.level)


class ImagePool(models.Model):
    pool_id = models.CharField(max_length=200, blank=True, null=True)
    caption = models.CharField(max_length=1000, blank=True, null=True)
    easy_pool = ArrayField(models.CharField(max_length=200), blank=True)
    medium_pool = ArrayField(models.CharField(max_length=200), blank=True)
    hard_pool = ArrayField(models.CharField(max_length=200), blank=True)
    obj = models.CharField(max_length=200, blank=True, null=True)
    target_image = models.CharField(max_length=200, blank=True, null=True)
    is_active = models.NullBooleanField(default=False, blank=True, null=True)


class Feedback(models.Model):
    understand_question = models.CharField(
        max_length=200, blank=True, null=True)
    understand_image = models.CharField(max_length=200, blank=True, null=True)
    fluency = models.CharField(max_length=200, blank=True, null=True)
    detail = models.CharField(max_length=200, blank=True, null=True)
    accurate = models.CharField(max_length=200, blank=True, null=True)
    consistent = models.CharField(max_length=200, blank=True, null=True)
    comments = models.CharField(max_length=200, blank=True, null=True)
    worker_id = models.CharField(max_length=100, blank=True, null=True)
    assignment_id = models.CharField(max_length=100, blank=True, null=True)
    level = models.CharField(max_length=100, blank=True, null=True)
    task = models.CharField(max_length=100, blank=True, null=True)
    hit_id = models.CharField(max_length=100, blank=True, null=True)
    game_id = models.CharField(max_length=100, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    bot = models.CharField(max_length=100, blank=True, null=True)
    is_active = models.NullBooleanField(default=True, blank=True, null=True)
