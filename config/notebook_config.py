# Databricks notebook source
# MAGIC %pip install lifelines

# COMMAND ----------

# Set config for database name, file paths, and table names
useremail = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
user = useremail.split('@')[0]
username_sql = user.replace(".", "_")
database_name = f'dm_ibm_telco_churn_{username_sql}'
data_path = '/home/{}/ibm-telco-churn'.format(user) 
_ = spark.sql(f'USE {database_name}')

# COMMAND ----------

import mlflow
experiment_name = f"/Users/{useremail}/survival"
mlflow.set_experiment(experiment_name) 

# COMMAND ----------

def tear_down():
  _ = sql("DROP DATABASE {} CASCADE".format(database_name))
  dbutils.fs.rm(data_path, True)
