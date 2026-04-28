# Generated migration for adding question_intent field to Message

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0006_ragquerylog_question_intent'),
    ]

    operations = [
        migrations.AddField(
            model_name='message',
            name='question_intent',
            field=models.CharField(
                blank=True,
                default='',
                help_text='Detected question intent for assistant responses (e.g., factual_lookup, team_comparison)',
                max_length=50,
            ),
        ),
    ]
