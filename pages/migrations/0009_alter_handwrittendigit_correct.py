# Generated by Django 3.2 on 2021-04-24 12:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('pages', '0008_alter_handwrittendigit_imagedisplay'),
    ]

    operations = [
        migrations.AlterField(
            model_name='handwrittendigit',
            name='correct',
            field=models.BooleanField(null=True),
        ),
    ]
