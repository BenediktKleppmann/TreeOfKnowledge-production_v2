# Generated by Django 2.0.8 on 2019-02-14 18:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('collection', '0014_auto_20190213_1411'),
    ]

    operations = [
        migrations.AddField(
            model_name='uploaded_dataset',
            name='list_of_matches',
            field=models.TextField(default='[null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null]'),
            preserve_default=False,
        ),
    ]
