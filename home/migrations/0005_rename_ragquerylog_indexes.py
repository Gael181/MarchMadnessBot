from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0004_rag_query_log'),
    ]

    operations = [
        migrations.RenameIndex(
            model_name='ragquerylog',
            new_name='home_ragque_created_44b93a_idx',
            old_name='home_ragquery_created_desc',
        ),
        migrations.RenameIndex(
            model_name='ragquerylog',
            new_name='home_ragque_outcome_c910bc_idx',
            old_name='home_ragquery_outcome_created',
        ),
    ]
