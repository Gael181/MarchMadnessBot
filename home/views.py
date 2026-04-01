from django.shortcuts import get_object_or_404, redirect, render
from .models import Chat, Message


def sidebar_context(selected_chat=None):
    return {
        'chats': Chat.objects.all(),
        'selected_chat': selected_chat,
    }


def home_view(request):
    context = sidebar_context()
    context['messages'] = []
    return render(request, 'home/chat.html', context)


def new_chat(request):
    chat = Chat.objects.create(title="New Chat")
    return redirect('chat_detail', chat_id=chat.id)


def chat_detail(request, chat_id):
    chat = get_object_or_404(Chat, id=chat_id)
    context = sidebar_context(chat)
    context['messages'] = chat.messages.all()
    return render(request, 'home/chat.html', context)


def save_message(request, chat_id):
    chat = get_object_or_404(Chat, id=chat_id)

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

    return redirect('chat_detail', chat_id=chat.id)


def rename_chat(request, chat_id):
    chat = get_object_or_404(Chat, id=chat_id)

    if request.method == 'POST':
        new_title = request.POST.get('title', '').strip()
        if new_title:
            chat.title = new_title[:120]
            chat.save()

    return redirect('chat_detail', chat_id=chat.id)