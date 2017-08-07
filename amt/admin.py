from django.contrib import admin

from .models import GameRound, ImageRanking, Feedback
from import_export import resources
from import_export.admin import ImportExportModelAdmin


class GameRoundResource(resources.ModelResource):

    class Meta:
        model = GameRound


class GameRoundAdmin(ImportExportModelAdmin):
    list_display = ('socket_id', 'user_picked_image', 'worker_id', 'assignment_id', 'level', 'task',
                    'hit_id', 'game_id', 'round_id', 'question', 'answer', 'target_image', 'created_at', 'bot',)
    list_filter = ('bot', 'worker_id', 'task', )
    search_fields = ['socket_id', 'user_picked_image', 'worker_id', 'assignment_id', 'level',
                     'hit_id', 'game_id', 'round_id', 'question', 'answer', 'target_image', 'created_at',  'bot', ]
    resource_class = GameRoundResource


class ImageRankingResource(resources.ModelResource):

    class Meta:
        model = ImageRanking


class ImageRankingAdmin(ImportExportModelAdmin):
    list_display = ('socket_id', 'final_image_list', 'worker_id', 'assignment_id', 'level',
                    'task', 'hit_id', 'game_id', 'target_image', 'created_at',  'bot', 'score', )
    list_filter = ('bot', 'worker_id', 'task', )
    search_fields = ['socket_id', 'final_image_list', 'worker_id', 'assignment_id',
                     'level', 'hit_id', 'game_id', 'target_image', 'created_at',  'bot', 'score', ]
    resource_class = ImageRankingResource


class FeedbackResource(resources.ModelResource):

    class Meta:
        model = Feedback


class FeedbackAdmin(ImportExportModelAdmin):
    list_display = ('hit_id', 'assignment_id', 'worker_id', 'understand_question', 'task', 'understand_image',
                    'fluency', 'detail', 'accurate', 'consistent', 'comments', 'level', 'game_id',  'bot',)
    list_filter = ('bot', 'worker_id', 'assignment_id', 'task', )
    resource_class = FeedbackResource

admin.site.register(GameRound, GameRoundAdmin)
admin.site.register(ImageRanking, ImageRankingAdmin)
admin.site.register(Feedback, FeedbackAdmin)
