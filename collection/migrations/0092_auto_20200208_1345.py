# Generated by Django 2.0.8 on 2020-02-08 12:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('collection', '0091_remove_simulation_model_just_learned_rules'),
    ]

    operations = [
        migrations.AddField(
            model_name='simulation_model',
            name='environment_end_time',
            field=models.BigIntegerField(default=1577836800),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='simulation_model',
            name='environment_start_time',
            field=models.BigIntegerField(default=946684800),
            preserve_default=False,
        ),
    ]
