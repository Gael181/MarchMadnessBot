from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import get_object_or_404, redirect, render

from .models import Chat, Message
from bot.services.chat_service import ChatService

from django.views.decorators.http import require_POST

def register_view(request):
    if request.user.is_authenticated:
        return redirect('home')

    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')
    else:
        form = UserCreationForm()

    return render(request, 'registration/register.html', {'form': form})

# Changed so that it's user-scoped instead cuz model belongs to user
def sidebar_context(user, selected_chat=None):
    return {
        'chats': Chat.objects.filter(user=user),
        'selected_chat': selected_chat,
    }

@login_required
def home_view(request):
    context = sidebar_context(request.user)
    context['messages'] = []
    return render(request, 'home/chat.html', context)

@login_required
def new_chat(request):
    chat = Chat.objects.create(user=request.user, title="New Chat")
    return redirect('chat_detail', chat_id=chat.id)

@login_required
def chat_detail(request, chat_id):
    chat = get_object_or_404(Chat, id=chat_id, user=request.user)
    context = sidebar_context(request.user, chat)
    context['messages'] = chat.messages.all()
    return render(request, 'home/chat.html', context)

@login_required
def save_message(request, chat_id):
    chat = get_object_or_404(Chat, id=chat_id, user=request.user)

    if request.method == 'POST':
        content = request.POST.get('content', '').strip()

        if content:
            Message.objects.create(
                chat=chat,
                role='user',
                content=content
            )

            if chat.title == "New Chat":
                chat.title = content[:40] + ("..." if len(content) > 40 else "")
                chat.save()

            assistant_response = ChatService.answer_question(
                content,
                temperature=chat.temperature,
            )

            Message.objects.create(
                chat=chat,
                role='assistant',
                content=assistant_response['text'],
                token_used=assistant_response['token_used'],
                response_time=assistant_response['response_time'],
            )

    return redirect('chat_detail', chat_id=chat.id)

@login_required
def rename_chat(request, chat_id):
    chat = get_object_or_404(Chat, id=chat_id, user=request.user)

    if request.method == 'POST':
        new_title = request.POST.get('title', '').strip()
        if new_title:
            chat.title = new_title[:120]
            chat.save()

    return redirect('chat_detail', chat_id=chat.id)


@login_required
@require_POST
def update_temperature(request, chat_id):
    chat = get_object_or_404(Chat, id=chat_id, user=request.user)

    value = request.POST.get("temperature", "0.2")

    try:
        temperature = float(value)
    except ValueError:
        temperature = 0.2

    temperature = max(0.0, min(2.0, temperature))
    chat.temperature = temperature
    chat.save()

    return redirect("chat_detail", chat_id=chat.id)