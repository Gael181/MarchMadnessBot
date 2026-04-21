# Generated migration for adding question_intent field to RagQueryLog

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0005_rename_ragquerylog_indexes'),
    ]

    operations = [
        migrations.AddField(
            model_name='ragquerylog',
            name='question_intent',
            field=models.CharField(
                blank=True,
                default='',
                help_text='Detected question intent (e.g., factual_lookup, team_comparison, trend_analysis)',
                max_length=50,
            ),
        ),
        migrations.AddIndex(
            model_name='ragquerylog',
            index=models.Index(
                fields=['question_intent', '-created_at'],
                name='home_ragque_questio_idx',
            ),
        ),
    ]
