# Covidsim imports

from covidsim_model.src.datasets import cases_death_process
from covidsim_model.src.datasets import country_metrics_process
from covidsim_model.src.datasets import demographic_process
from covidsim_model.src.datasets import gmobility_process
from covidsim_model.src.datasets import oxford_process
import pandas as pd
import os
# [START import_module]
from datetime import datetime, timedelta
from textwrap import dedent

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

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

def _get_csv_as_dataframe(input_base, name):
    return pd.read_csv(os.path.join(input_base, name + '.csv'), low_memory=False)

def merge_results():
    oxford = _get_csv_as_dataframe(_data_root, 'oxford')
    gmobility = _get_csv_as_dataframe(_data_root, 'gmobility')
    demographic = _get_csv_as_dataframe(_data_root, 'demographic')
 #   rt_estimation = _get_csv_as_dataframe(_data_root, 'rt_estimation')
    country_metrics = _get_csv_as_dataframe(_data_root, 'country_metrics')

    out = pd.merge(oxford, gmobility, how="inner", on=["CountryName", "Date"])
#    out = pd.merge(out, rt_estimation, how="inner", on=["CountryName", "Date"])
    out = pd.merge(out, demographic, how="inner", on=["CountryName"])
    out = pd.merge(out, country_metrics, how="inner", on=["CountryName"])

    merged_data_dir = os.path.join(_data_root, 'merged')

    if not os.path.exists(merged_data_dir):
        os.mkdir(merged_data_dir)

    out.to_csv(os.path.join(merged_data_dir, 'data.csv'), index=False)



with DAG(
    'covidsim_etl',
    default_args=default_args,
    description='DAG containing the ETL operation for the covidsim model',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=['covidsim','ETL'],
) as dag:
    # [END instantiate_dag]

    t1 = PythonOperator(
            task_id = "case_death_computation",
            python_callable = cases_death_process.run,
            op_kwargs = {'config_path':'/Users/adriano.franci/projects/idoml/airflow/dags/covidsim_model/config/config.yaml'}
            )
    t2 = PythonOperator(
            python_callable = country_metrics_process.run,
            task_id = "country_metrics_process",
            op_kwargs = {'config_path':'/Users/adriano.franci/projects/idoml/airflow/dags/covidsim_model/config/config.yaml'}
            )
    t3 = PythonOperator(
            python_callable = demographic_process.run,
            task_id = "demographic_process",
            op_kwargs = {'config_path':'/Users/adriano.franci/projects/idoml/airflow/dags/covidsim_model/config/config.yaml'}
            )
    t4 = PythonOperator(
            python_callable = gmobility_process.run,
            task_id = "gmobility_process",
            op_kwargs = {'config_path':'/Users/adriano.franci/projects/idoml/airflow/dags/covidsim_model/config/config.yaml'}
            )
    t5 = PythonOperator(
            python_callable = oxford_process.run,
            task_id = "oxford_process",
            op_kwargs = {'config_path':'/Users/adriano.franci/projects/idoml/airflow/dags/covidsim_model/config/config.yaml'}
            )
    t6 = PythonOperator(
            python_callable = merge_results,
            task_id = "fat_merge",
            op_kwargs = {'config_path':'/Users/adriano.franci/projects/idoml/airflow/dags/covidsim_model/config/config.yaml'}
            )
    
    t6.set_upstream([t1,t2,t3,t4,t5])
