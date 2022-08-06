# Databricks notebook source
# MAGIC %md This notebook sets up the companion cluster(s) to run the solution accelerator. It also creates the Workflow to create a Workflow DAG and illustrate the order of execution. Feel free to interactively run notebooks with the cluster or to run the Workflow to see how this solution accelerator executes. Happy exploring!
# MAGIC 
# MAGIC The pipelines, workflows and clusters created in this script are user-specific, so you can alter the workflow and cluster via UI without affecting other users. Running this script again after modification resets them.
# MAGIC 
# MAGIC **Note**: If the job execution fails, please confirm that you have set up other environment dependencies as specified in the accelerator notebooks. Accelerators sometimes require the user to set up additional cloud infra or data access, for instance. 

# COMMAND ----------

# DBTITLE 0,Install util packages
# MAGIC %pip install git+https://github.com/databricks-academy/dbacademy-rest git+https://github.com/databricks-academy/dbacademy-gems git+https://github.com/databricks-industry-solutions/notebook-solution-companion

# COMMAND ----------

from solacc.companion import NotebookSolutionCompanion

# COMMAND ----------

job_json = {
        "timeout_seconds": 7200,
        "max_concurrent_runs": 1,
        "tags": {
            "usage": "solacc_testing",
            "group": "CME"
        },
        "tasks": [
            {
                "job_cluster_key": "survival_cluster",
                "libraries": [],
                "notebook_task": {
                    "notebook_path": f"01_intro"
                },
                "task_key": "survival_01",
                "description": ""
            },
            {
                "job_cluster_key": "survival_cluster",
                "libraries": [],
                "notebook_task": {
                    "notebook_path": f"02_kaplan_meier"
                },
                "task_key": "survival_02",
                "description": "",
                "depends_on": [
                    {
                        "task_key": "survival_01"
                    }
                ]
            },
            {
                "job_cluster_key": "survival_cluster",
                "libraries": [],
                "notebook_task": {
                    "notebook_path": f"03_cox_proportional_hazards"
                },
                "task_key": "survival_03",
                "description": "",
                "depends_on": [
                    {
                        "task_key": "survival_02"
                    }
                ]
            },
            {
                "job_cluster_key": "survival_cluster",
                "libraries": [],
                "notebook_task": {
                    "notebook_path": f"04_accelerated_failure_time"
                },
                "task_key": "survival_04",
                "description": "",
                "depends_on": [
                    {
                        "task_key": "survival_03"
                    }
                ]
            },
            {
                "job_cluster_key": "survival_cluster",
                "libraries": [],
                "notebook_task": {
                    "notebook_path": f"05_customer_lifetime_value"
                },
                "task_key": "survival_05",
                "description": "",
                "depends_on": [
                    {
                        "task_key": "survival_04"
                    }
                ]
            },
            {
                "job_cluster_key": "survival_cluster",
                "libraries": [],
                "notebook_task": {
                    "notebook_path": f"06_conclusion_and_references"
                },
                "task_key": "survival_06",
                "description": "",
                "depends_on": [
                    {
                        "task_key": "survival_05"
                    }
                ]
            }
        ],
        "job_clusters": [
            {
                "job_cluster_key": "survival_cluster",
                "new_cluster": {
                    "spark_version": "10.4.x-cpu-ml-scala2.12",
                "spark_conf": {
                    "spark.databricks.delta.formatCheck.enabled": "false"
                    },
                    "num_workers": 8,
                    "node_type_id": {"AWS": "i3.xlarge", "MSA": "Standard_D3_v2", "GCP": "n1-highmem-4"}, # different from standard API
                    "custom_tags": {
                        "usage": "solacc_testing"
                    },
                }
            }
        ]
    }


# COMMAND ----------

NotebookSolutionCompanion().deploy_compute(job_json)

# COMMAND ----------


