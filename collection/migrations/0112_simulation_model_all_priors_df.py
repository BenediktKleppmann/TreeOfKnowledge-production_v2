# Generated by Django 2.0.8 on 2020-08-21 16:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('collection', '0111_logged_variable'),
    ]

    operations = [
        migrations.AddField(
            model_name='simulation_model',
            name='all_priors_df',
            field=models.TextField(default='{}'),
            preserve_default=False,
        ),
    ]