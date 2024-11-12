# Generated by Django 2.0.8 on 2019-05-15 09:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('collection', '0053_auto_20190515_1059'),
    ]

    operations = [
        migrations.AlterField(
            model_name='data_point',
            name='attribute_id',
            field=models.TextField(),
        ),
        migrations.AlterField(
            model_name='data_point',
            name='object_id',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='data_point',
            name='valid_time_end',
            field=models.IntegerField(default=1262304000),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='data_point',
            name='valid_time_start',
            field=models.IntegerField(default=1104537600),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='data_point',
            name='value_as_string',
            field=models.TextField(),
        ),
    ]
