# Generated by Django 2.0.8 on 2019-11-06 13:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('collection', '0077_auto_20191105_1135'),
    ]

    operations = [
        migrations.AlterField(
            model_name='attribute',
            name='expected_valid_period',
            field=models.BigIntegerField(),
        ),
    ]
