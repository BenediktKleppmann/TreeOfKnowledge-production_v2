# Generated by Django 2.0.8 on 2019-02-12 12:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('collection', '0010_profile'),
    ]

    operations = [
        migrations.AddField(
            model_name='uploaded_dataset',
            name='attribute_selection',
            field=models.TextField(default='[]'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='uploaded_dataset',
            name='datetime_column',
            field=models.TextField(default='[]'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='uploaded_dataset',
            name='object_identifiers',
            field=models.TextField(default='[]'),
            preserve_default=False,
        ),
    ]
