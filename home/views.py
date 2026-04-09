from datetime import timedelta

from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.contrib.auth.forms import UserCreationForm
from django.db.models import Avg, Count
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from .models import Chat, Message, RagQueryLog
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

            answer = ChatService.answer_question(
                content,
                temperature=chat.temperature,
            )

            RagQueryLog.objects.create(
                user=request.user,
                latency_ms=max(0, min(2_147_483_647, int(round(answer.latency_ms)))),
                outcome=answer.outcome,
                error_message=answer.error_message or '',
            )

            Message.objects.create(
                chat=chat,
                role='assistant',
                content=answer.text,
                token_used=answer.token_used,
                response_time=answer.response_time,
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


def _percentile(sorted_vals, p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[int(k)])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


@login_required
def admin_portal(request):
    """Staff analytics: usage logs (US12) + performance / reliability (US10)."""
    if not request.user.is_staff:
        raise PermissionDenied

    now = timezone.now()
    range_key = request.GET.get('range', '24h')
    if range_key == '7d':
        cutoff = now - timedelta(days=7)
        window_label = 'Last 7 days'
        window_hours = 24 * 7
    else:
        cutoff = now - timedelta(hours=24)
        window_label = 'Last 24 hours'
        window_hours = 24
        range_key = '24h'

    recent_qs = RagQueryLog.objects.filter(created_at__gte=cutoff)
    agg = recent_qs.aggregate(
        n=Count('id'),
        avg_latency=Avg('latency_ms'),
    )
    by_outcome = dict(
        recent_qs.values('outcome').annotate(c=Count('id')).values_list('outcome', 'c')
    )

    latencies = sorted(recent_qs.values_list('latency_ms', flat=True))
    p50 = _percentile(latencies, 0.50)
    p95 = _percentile(latencies, 0.95)

    failures = sum(
        by_outcome.get(o, 0)
        for o in ('error', 'llm_fallback', 'no_results')
    )
    success_n = by_outcome.get('success', 0)
    failure_rate = (failures / agg['n']) if agg['n'] else 0.0

    total_n = agg['n'] or 0
    outcome_bars = []
    for key in ('success', 'llm_fallback', 'no_results', 'error'):
        c = by_outcome.get(key, 0)
        pct = round(100.0 * c / total_n, 1) if total_n else 0.0
        outcome_bars.append({'key': key, 'count': c, 'pct': pct})

    unique_query_users = (
        recent_qs.exclude(user__isnull=True).values('user').distinct().count()
    )
    new_chats = Chat.objects.filter(created_at__gte=cutoff).count()
    user_messages = Message.objects.filter(
        created_at__gte=cutoff,
        role='user',
    ).count()

    table_logs = (
        RagQueryLog.objects.filter(created_at__gte=cutoff)
        .select_related('user')
        .order_by('-created_at')[:250]
    )

    context = {
        'range_key': range_key,
        'window_label': window_label,
        'window_hours': window_hours,
        'total_queries': total_n,
        'unique_query_users': unique_query_users,
        'new_chats': new_chats,
        'user_messages': user_messages,
        'avg_latency_ms': round(agg['avg_latency'] or 0, 1),
        'p50_ms': round(p50, 1),
        'p95_ms': round(p95, 1),
        'by_outcome': by_outcome,
        'outcome_bars': outcome_bars,
        'failure_rate_pct': round(100.0 * failure_rate, 1),
        'success_n': success_n,
        'failures_n': failures,
        'recent_logs': table_logs,
    }
    return render(request, 'home/admin_portal.html', context)