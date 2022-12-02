# Generated by Django 4.1.3 on 2022-12-02 09:11

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ("users", "0002_prediccion"),
    ]

    operations = [
        migrations.AddField(
            model_name="prediccion",
            name="created_at",
            field=models.DateTimeField(
                auto_now_add=True, default=django.utils.timezone.now
            ),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="prediccion", name="pred", field=models.FloatField(default=1),
        ),
        migrations.AddField(
            model_name="prediccion",
            name="updated_at",
            field=models.DateTimeField(auto_now=True),
        ),
    ]