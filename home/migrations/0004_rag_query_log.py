# RagQueryLog model (admin analytics). Depends on main's message token fields migration.

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('home', '0003_message_response_time_message_token_used'),
    ]

    operations = [
        migrations.CreateModel(
            name='RagQueryLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('latency_ms', models.PositiveIntegerField()),
                ('outcome', models.CharField(max_length=32)),
                ('error_message', models.TextField(blank=True)),
                ('user', models.ForeignKey(
                    blank=True,
                    null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    related_name='rag_query_logs',
                    to=settings.AUTH_USER_MODEL,
                )),
            ],
            options={
                'ordering': ['-created_at'],
            },
        ),
        migrations.AddIndex(
            model_name='ragquerylog',
            index=models.Index(fields=['-created_at'], name='home_ragquery_created_desc'),
        ),
        migrations.AddIndex(
            model_name='ragquerylog',
            index=models.Index(fields=['outcome', '-created_at'], name='home_ragquery_outcome_created'),
        ),
    ]
