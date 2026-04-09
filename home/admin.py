from django.contrib import admin
from .models import Chat, Message, RagQueryLog

admin.site.register(Chat)
admin.site.register(Message)


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