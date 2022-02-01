# Generated by Django 4.0 on 2022-01-18 19:28

import backend.fields
from django.db import migrations, models
import django.db.models.deletion
import uuid


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='EstimationTask',
            fields=[
                ('uid', models.UUIDField(default=uuid.uuid4, editable=False, help_text='A unique identifier.', primary_key=True, serialize=False, unique=True, verbose_name='unique identifier')),
                ('iter_counter', models.PositiveIntegerField(default=0, help_text='An algorithm iteration counter', verbose_name='iteration counter')),
                ('eval_counter', models.PositiveIntegerField(default=0, help_text='An objective function evaluation counter', verbose_name='function evaluation counter')),
            ],
            options={
                'verbose_name': 'Estimation Task',
                'verbose_name_plural': 'Estimation Tasks',
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='FileIO',
            fields=[
                ('uid', models.UUIDField(default=uuid.uuid4, editable=False, help_text='A unique identifier.', primary_key=True, serialize=False, unique=True, verbose_name='unique identifier')),
                ('output_directory_identifier', models.CharField(blank=True, default='', max_length=250, null=True, verbose_name='')),
                ('date_time', models.DateTimeField(auto_now=True, help_text='The date and time of creation.', null=True, verbose_name='date time')),
                ('root_input_directory', models.CharField(blank=True, default='', max_length=250, null=True, verbose_name='')),
                ('directory_counter', models.PositiveIntegerField(default=0, verbose_name='')),
            ],
            options={
                'verbose_name': 'FileIO',
                'verbose_name_plural': 'FileIO',
                'ordering': ['date_time'],
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='Iteration',
            fields=[
                ('uid', models.UUIDField(default=uuid.uuid4, editable=False, help_text='A unique identifier.', primary_key=True, serialize=False, unique=True, verbose_name='unique identifier')),
                ('iter_number', models.PositiveIntegerField(help_text='The current iteration.', null=True, verbose_name='iteration number')),
                ('x_movement', models.FloatField(help_text="The progress made at the current iteration: ||x - x'|| /             max(1, ||x||).", null=True, verbose_name='x movement')),
                ('of_saturation', models.FloatField(help_text="The progress made at the current iteration: |F(x) - F(x')| /             max(1, |F(x)|).", null=True, verbose_name='objective function saturation')),
                ('direction_magnitude', models.FloatField(help_text='The progress made at the current iteration: ||grad(F(x))||.', null=True, verbose_name='direction mangnitude')),
                ('estimation_task', models.ForeignKey(blank=True, help_text='All data pertaining to a particular estimation task.', null=True, on_delete=django.db.models.deletion.CASCADE, related_name='iterations', to='backend.estimationtask', verbose_name='estimation task')),
            ],
            options={
                'verbose_name': 'Iteration',
                'verbose_name_plural': 'Iterations',
                'ordering': ['iter_number'],
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='ScenarioData',
            fields=[
                ('uid', models.UUIDField(default=uuid.uuid4, editable=False, help_text='A unique identifier.', primary_key=True, serialize=False, unique=True, verbose_name='unique identifier')),
                ('random_seed', models.IntegerField(editable=False, verbose_name='')),
                ('interval_start', models.PositiveIntegerField(editable=False, verbose_name='')),
                ('interval_end', models.PositiveIntegerField(editable=False, verbose_name='')),
                ('aggregation_intervals', models.PositiveIntegerField(editable=False, verbose_name='')),
                ('interval_increment', models.IntegerField(default=None, editable=False, null=True, verbose_name='')),
            ],
            options={
                'verbose_name': 'Scenario Data',
                'verbose_name_plural': 'Scenario Data',
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='SumoNetwork',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('network_files', backend.fields.JSONField(blank=True, default=None, null=True, verbose_name='')),
            ],
        ),
        migrations.CreateModel(
            name='SumoMesoSimulationRun',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('simulation_files', backend.fields.JSONField(blank=True, null=True, verbose_name='')),
                ('simulator_options', backend.fields.JSONField(verbose_name='')),
                ('wall_time', models.FloatField(help_text='The wall clock time it took to run the simulator', verbose_name='wall clock time')),
                ('sysusr_time', models.FloatField(help_text='The system+user time it took to evaluate the simulator', verbose_name='system+user time')),
                ('file_io', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='sumo_meso_simulation_runs', to='backend.fileio', verbose_name='')),
                ('scenario_data', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='sumo_meso_simulation_runs', to='backend.scenariodata', verbose_name='')),
            ],
        ),
        migrations.AddField(
            model_name='scenariodata',
            name='network',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='scenario_data', to='backend.sumonetwork', verbose_name=''),
        ),
        migrations.CreateModel(
            name='ObjectiveFunctionEvaluation',
            fields=[
                ('uid', models.UUIDField(default=uuid.uuid4, editable=False, help_text='A unique identifier.', primary_key=True, serialize=False, unique=True, verbose_name='unique identifier')),
                ('of', backend.fields.JSONField(help_text='A dictionary of one or more objective function values.', verbose_name='objective function values')),
                ('simulation_data', backend.fields.JSONField(verbose_name='')),
                ('eval_number', models.PositiveIntegerField(help_text='The current objective function evaluation number.', verbose_name='function evaluation number.')),
                ('iter_number', models.PositiveIntegerField(help_text='The current iteration number.', null=True, verbose_name='iteration number')),
                ('wall_time', models.FloatField(help_text='The wall clock time it took to evaluate the objective function.', verbose_name='wall clock time')),
                ('sysusr_time', models.FloatField(help_text='The system+user time it took to evaluate the objective function.', verbose_name='system+user time')),
                ('take_step', models.BooleanField(default=False, help_text='A value indicating if the objective function evaluation was made             at the very end of an iteration. That is, the value indicates if the             objective function value is obtained from the adjusted solution: x_new             = x_old -gradient * step-length.', verbose_name='take step')),
                ('estimation_task', models.ForeignKey(blank=True, help_text='All data pertaining to a particular estimation task.', null=True, on_delete=django.db.models.deletion.CASCADE, related_name='of_evals', to='backend.estimationtask', verbose_name='objective function evaluations')),
                ('iteration', models.ForeignKey(blank=True, help_text='All data pertaining to a particular iteration of the alg.', null=True, on_delete=django.db.models.deletion.CASCADE, related_name='iteration_of_evaluations', to='backend.iteration', verbose_name='iteration')),
            ],
            options={
                'verbose_name': 'Objective Function Evaluation',
                'verbose_name_plural': 'Objective Function Evaluations',
                'ordering': ['iter_number'],
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='EstimationTaskMetaData',
            fields=[
                ('uid', models.UUIDField(default=uuid.uuid4, editable=False, help_text='A unique identifier.', primary_key=True, serialize=False, unique=True, verbose_name='unique identifier')),
                ('task_id', models.CharField(blank=True, default='', help_text='A user-defined identifier.', max_length=250, null=True, verbose_name='task id')),
                ('date_time', models.DateTimeField(auto_now=True, help_text='The date and time of creation.', null=True, verbose_name='date time')),
                ('file_io', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='file_io', to='backend.fileio', verbose_name='')),
                ('scenario_data', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='scenario_data', to='backend.scenariodata', verbose_name='')),
            ],
            options={
                'verbose_name': 'Estimation Task Meta Data',
                'verbose_name_plural': 'Estimation Task Meta Data',
                'ordering': ['date_time'],
                'abstract': False,
            },
        ),
        migrations.AddField(
            model_name='estimationtask',
            name='metadata',
            field=models.ForeignKey(blank=True, help_text='Metadata about a particular estimation task.', null=True, on_delete=django.db.models.deletion.CASCADE, related_name='estimation_task', to='backend.estimationtaskmetadata', verbose_name='metadata'),
        ),
    ]
