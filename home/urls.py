from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),

    path(
        'login/',
        auth_views.LoginView.as_view(
            template_name='registration/login.html',
            redirect_authenticated_user=True
        ),
        name='login'
    ),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('register/', views.register_view, name='register'),

    path('chat/new/', views.new_chat, name='new_chat'),
    path('chat/<int:chat_id>/', views.chat_detail, name='chat_detail'),
    path('chat/<int:chat_id>/send/', views.save_message, name='save_message'),
    path('chat/<int:chat_id>/rename/', views.rename_chat, name='rename_chat'),
    path("chat/<int:chat_id>/temperature/", views.update_temperature, name="update_temperature"),
]