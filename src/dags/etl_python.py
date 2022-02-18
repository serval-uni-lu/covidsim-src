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
    'Covidsim_DAG_python',
    default_args=default_args,
    description='DAG containing the ETL operation for the covidsim model',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=['covidsim','dvc'],
) as dag:
    # [END instantiate_dag]
    def dvc_pull_callable():
        from dvc.repo import Repo
        repo = Repo(f"{os.getenv('AIRFLOW_HOME')}/dags/{data_repo}")
        repo.pull()

    def data_process(script):
        os.chdir(f"{os.getenv('AIRFLOW_HOME')}/dags/{src_repo}")
        import sys,importlib
        sys.path.append(f"{os.getenv('AIRFLOW_HOME')}/dags/{src_repo}")
        module = importlib.import_module(f"src.{script}")
        getattr(module,'run')()

    dvcp_pull = PythonOperator(
            task_id = "dvc_pull",
            python_callable = dvc_pull_callable)
    
    cd_process = PythonOperator(
            task_id = "case_death_process",
            python_callable = data_process,
            op_args = ["datasets.cases_death_process"])

    gmobility_process = PythonOperator(
            task_id = "gmobility_process",
            python_callable = data_process,
            op_args = ["datasets.gmobility_process"])

    country_metrics_process = PythonOperator(
            task_id = "country_metrics_process",
            python_callable = data_process,
            op_args = ["datasets.country_metrics_process"])

    oxford_process = PythonOperator(
            task_id = "oxford_process",
            python_callable = data_process,
            op_args = ["datasets.oxford_process"])

    demographic_process = PythonOperator(
            task_id = "demographic_process",
            python_callable = data_process,
            op_args = ["datasets.demographic_process"])

    rt_estimate = PythonOperator(
            task_id = "rt_estimate",
            python_callable = data_process,
            op_args = ['estimate_rt']
            )
    
    merge_datasets = PythonOperator(
            task_id = "merge_datasets",
            python_callable = data_process,
            op_args = ['create_dataset'])

    dvcp_pull >> [gmobility_process,oxford_process,demographic_process,country_metrics_process] >> merge_datasets
    dvcp_pull >> cd_process >> rt_estimate >> merge_datasets 
