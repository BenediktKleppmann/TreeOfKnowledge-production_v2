# Generated by Django 2.0.8 on 2019-03-18 08:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('collection', '0025_object_types_object_icon'),
    ]

    operations = [
        migrations.AddField(
            model_name='attribute',
            name='behaviour_rules',
            field=models.TextField(default='[]'),
            preserve_default=False,
        ),
    ]
