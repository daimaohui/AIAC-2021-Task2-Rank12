# License: MIT

# Generated by Django 3.1.4 on 2020-12-18 04:59

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='OnlineOptimizer',
            fields=[
                ('id', models.CharField(max_length=100, primary_key=True, serialize=False)),
                ('pub_date', models.DateTimeField(verbose_name='date published')),
            ],
        ),
        migrations.CreateModel(
            name='FinishedRun',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('config', models.CharField(max_length=200)),
                ('perf', models.FloatField()),
                ('host_optimizer', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='bo_advice.onlineoptimizer')),
            ],
        ),
    ]
