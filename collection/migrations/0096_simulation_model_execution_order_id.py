# Generated by Django 2.0.8 on 2020-03-02 12:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('collection', '0095_execution_order'),
    ]

    operations = [
        migrations.AddField(
            model_name='simulation_model',
            name='execution_order_id',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
    ]
