from django.contrib import admin
from .models import Chat, Message, RagQueryLog


class ChatAdmin(admin.ModelAdmin):
    list_display = ('title', 'created_at', 'updated_at')
    search_fields = ('title',)
    ordering = ('-updated_at',)


class MessageAdmin(admin.ModelAdmin):
    list_display = ('truncated_content', 'role', 'created_at', 'token_used', 'response_time')
    search_fields = ('content', 'role')
    ordering = ('-created_at',)

    def truncated_content(self, obj):
        return obj.content[:50] + '...' if len(obj.content) > 50 else obj.content


admin.site.register(Chat, ChatAdmin)
admin.site.register(Message, MessageAdmin)


@admin.register(RagQueryLog)
class RagQueryLogAdmin(admin.ModelAdmin):
    list_display = ('created_at', 'user', 'latency_ms', 'outcome')
    list_filter = ('outcome',)
    readonly_fields = ('created_at', 'user', 'latency_ms', 'outcome', 'error_message')
    ordering = ('-created_at',)

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False
