# Generated by Django 2.0.8 on 2019-04-12 18:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('collection', '0037_simulation_model_simulation_start_time'),
    ]

    operations = [
        migrations.AddField(
            model_name='simulation_model',
            name='linegraph_data',
            field=models.TextField(null=True),
        ),
    ]
