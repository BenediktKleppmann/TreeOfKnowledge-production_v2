# Generated by Django 2.0.8 on 2020-01-27 13:28

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('collection', '0086_auto_20200127_1423'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='rule_parameter',
            name='list_of_probabilities',
        ),
        migrations.RemoveField(
            model_name='rule_parameter',
            name='nb_of_sim_in_which_rule_was_used',
        ),
        migrations.RemoveField(
            model_name='rule_parameter',
            name='nb_of_simulations',
        ),
        migrations.RemoveField(
            model_name='rule_parameter',
            name='nb_of_values_in_posterior',
        ),
    ]