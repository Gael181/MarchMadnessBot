from django.db import models
from django.contrib.auth.models import User


class Chat(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chats')
    title = models.CharField(max_length=120, default="New Chat")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    temperature = models.FloatField(default=0.2)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return self.title


class Message(models.Model):
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
    ]

    chat = models.ForeignKey(Chat, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='user')
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.role}: {self.content[:40]}"


class RagQueryLog(models.Model):
    """One row per chat turn that invoked retrieval + generation (User Story 10 / 12)."""

    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='rag_query_logs')
    created_at = models.DateTimeField(auto_now_add=True)
    latency_ms = models.PositiveIntegerField()
    outcome = models.CharField(max_length=32)
    error_message = models.TextField(blank=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at']),
            models.Index(fields=['outcome', '-created_at']),
        ]

    def __str__(self):
        return f"{self.created_at} {self.outcome} {self.latency_ms}ms"