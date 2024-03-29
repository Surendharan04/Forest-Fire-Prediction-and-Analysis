# Generated by Django 4.1.4 on 2023-03-21 14:35

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("new", "0001_initial"),
    ]

    operations = [
        migrations.RenameField(
            model_name="userpredictdatamodel",
            old_name="ADV_R",
            new_name="BUI",
        ),
        migrations.RenameField(
            model_name="userpredictdatamodel",
            old_name="ADV_S",
            new_name="Classes",
        ),
        migrations.RenameField(
            model_name="userpredictdatamodel",
            old_name="Attack_type",
            new_name="DC",
        ),
        migrations.RenameField(
            model_name="userpredictdatamodel",
            old_name="DATA_R",
            new_name="DMC",
        ),
        migrations.RenameField(
            model_name="userpredictdatamodel",
            old_name="DATA_S",
            new_name="FFMC",
        ),
        migrations.RenameField(
            model_name="userpredictdatamodel",
            old_name="Data_Sent_To_BS",
            new_name="FWI",
        ),
        migrations.RenameField(
            model_name="userpredictdatamodel",
            old_name="Dist_To_CH",
            new_name="ISI",
        ),
        migrations.RenameField(
            model_name="userpredictdatamodel",
            old_name="Expaned_Energy",
            new_name="RH",
        ),
        migrations.RenameField(
            model_name="userpredictdatamodel",
            old_name="Is_CH",
            new_name="Rain",
        ),
        migrations.RenameField(
            model_name="userpredictdatamodel",
            old_name="JOIN_R",
            new_name="Temperature",
        ),
        migrations.RenameField(
            model_name="userpredictdatamodel",
            old_name="JOIN_S",
            new_name="Ws",
        ),
        migrations.RemoveField(
            model_name="userpredictdatamodel",
            name="Rank",
        ),
        migrations.RemoveField(
            model_name="userpredictdatamodel",
            name="SCH_R",
        ),
        migrations.RemoveField(
            model_name="userpredictdatamodel",
            name="SCH_S",
        ),
        migrations.RemoveField(
            model_name="userpredictdatamodel",
            name="Time",
        ),
        migrations.RemoveField(
            model_name="userpredictdatamodel",
            name="dist_CH_To_BS",
        ),
        migrations.RemoveField(
            model_name="userpredictdatamodel",
            name="send_code",
        ),
        migrations.RemoveField(
            model_name="userpredictdatamodel",
            name="who_CH",
        ),
    ]
