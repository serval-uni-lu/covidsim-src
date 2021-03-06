# Covidsim imports

import pandas as pd
import os
# [START import_module]
from datetime import datetime, timedelta
from textwrap import dedent
from docker.types import Mount
# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator

# [END import_module]

# [START default_args]
# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}
# [END default_args]

# [START instantiate_dag]

_data_root = "/Users/adriano.franci/projects/idoml/airflow/dags/covidsim_model/data/processed"
dags_path = "/home/idoml/airflow/dags"
data_repo = "covidsim-data"
src_repo = "covidsim-src"
with DAG(
    'Covidsim_DAG',
    default_args=default_args,
    description='DAG containing the ETL operation for the covidsim model',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=['covidsim','dvc'],
) as dag:
    # [END instantiate_dag]
    dvc_pull = DockerOperator(
            task_id = "dvc_pull",
            image = "covidsim:latest",
            api_version = "auto",
            command = "sh script.sh ",# pitfall of airflow, should contain a space at the end
            auto_remove=True,
            network_mode='host',
            working_dir=f"/{data_repo}",
            mounts=[Mount(target=f"/{data_repo}",source=f"{dags_path}/{data_repo}",type='bind')]
            )
    oxford_process = DockerOperator(
            task_id = "oxford_process",
            image = "covidsim:latest",
            api_version = "auto",
            command = "sh data_process.sh oxford ",# pitfall of airflow, should contain a space at the end
            auto_remove=True,
            network_mode='host',
            working_dir=f"/{src_repo}",
            mounts=[Mount(target=f"/{src_repo}",source=f"{dags_path}/{src_repo}",type='bind'),Mount(target=f"/{data_repo}",source=f"{dags_path}/{data_repo}",type='bind')]
            )
    cases_death_process = DockerOperator(
            task_id = "cases_death_process",
            image = "covidsim:latest",
            api_version = "auto",
            command = "sh data_process.sh cases_death ",# pitfall of airflow, should contain a space at the end
            auto_remove=True,
            network_mode='host',
            working_dir=f"/{src_repo}",
            mounts=[Mount(target=f"/{src_repo}",source=f"{dags_path}/{src_repo}",type='bind'),Mount(target=f"/{data_repo}",source=f"{dags_path}/{data_repo}",type='bind')]
            )
    gmobility_process = DockerOperator(
            task_id = "gmobility_process",
            image = "covidsim:latest",
            api_version = "auto",
            command = "sh data_process.sh gmobility ",# pitfall of airflow, should contain a space at the end
            auto_remove=True,
            network_mode='host',
            working_dir=f"/{src_repo}",
            mounts=[Mount(target=f"/{src_repo}",source=f"{dags_path}/{src_repo}",type='bind'),Mount(target=f"/{data_repo}",source=f"{dags_path}/{data_repo}",type='bind')]
            )
    demographic_process = DockerOperator(
            task_id = "demographic_process",
            image = "covidsim:latest",
            api_version = "auto",
            command = "sh data_process.sh demographic ",# pitfall of airflow, should contain a space at the end
            auto_remove=True,
            network_mode='host',
            working_dir=f"/{src_repo}",
            mounts=[Mount(target=f"/{src_repo}",source=f"{dags_path}/{src_repo}",type='bind'),Mount(target=f"/{data_repo}",source=f"{dags_path}/{data_repo}",type='bind')]
            )

    country_metrics_process = DockerOperator(
            task_id = "country_metrics_process",
            image = "covidsim:latest",
            api_version = "auto",
            command = "sh data_process.sh country_metrics ",# pitfall of airflow, should contain a space at the end
            auto_remove=True,
            network_mode='host',
            working_dir=f"/{src_repo}",
            mounts=[Mount(target=f"/{src_repo}",source=f"{dags_path}/{src_repo}",type='bind'),Mount(target=f"/{data_repo}",source=f"{dags_path}/{data_repo}",type='bind')]
            )

    estimate_rt= DockerOperator(
            task_id = "estimate_rt",
            image = "covidsim:latest",
            api_version = "auto",
            command = "sh rt_process.sh ",# pitfall of airflow, should contain a space at the end
            auto_remove=True,
            network_mode='host',
            working_dir=f"/{src_repo}",
            mounts=[Mount(target=f"/{src_repo}",source=f"{dags_path}/{src_repo}",type='bind'),Mount(target=f"/{data_repo}",source=f"{dags_path}/{data_repo}",type='bind')]
            )
    merge_outputs= DockerOperator(
            task_id = "merge_outputs",
            image = "covidsim:latest",
            api_version = "auto",
            command = "sh merge_process.sh ",# pitfall of airflow, should contain a space at the end
            auto_remove=True,
            network_mode='host',
            working_dir=f"/{src_repo}",
            mounts=[Mount(target=f"/{src_repo}",source=f"{dags_path}/{src_repo}",type='bind'),Mount(target=f"/{data_repo}",source=f"{dags_path}/{data_repo}",type='bind')]
            )

    dvc_pull >> [oxford_process,country_metrics_process,demographic_process,gmobility_process] >> merge_outputs
    dvc_pull >> cases_death_process >> estimate_rt >> merge_outputs
