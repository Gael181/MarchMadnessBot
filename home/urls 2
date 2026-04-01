from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('chat/new/', views.new_chat, name='new_chat'),
    path('chat/<int:chat_id>/', views.chat_detail, name='chat_detail'),
    path('chat/<int:chat_id>/send/', views.save_message, name='save_message'),
    path('chat/<int:chat_id>/rename/', views.rename_chat, name='rename_chat'),
]