# Generated by Django 3.2 on 2021-04-17 05:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('pages', '0002_handwrittendigit_guess'),
    ]

    operations = [
        migrations.AlterField(
            model_name='handwrittendigit',
            name='guess',
            field=models.CharField(max_length=2),
        ),
    ]
