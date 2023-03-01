# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/survival-analysis. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/survival-analysis-for-churn-and-lifetime-value.

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="text-align: left">
# MAGIC   <img src="https://brysmiwasb.blob.core.windows.net/demos/images/ME_solution-accelerator.png"; width="50%">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Overview
# MAGIC 
# MAGIC * Survival Analysis is a collection of statistical methods used to examine and predict the time until an event of interest occurs. This form of analysis originated in Healthcare, with a focus on time-to-death. Since then, Survival Analysis has been successfully applied to use cases in virtually every industry around the globe. 
# MAGIC 
# MAGIC * Telco Use Case Examples
# MAGIC   * `Customer Retention:` It is widely accepted that the cost of retention is lower than the cost of acquisition.  With the event of interest being a service cancellation, Telco companies can more effectively manage churn by using Survival Analysis to better predict at what point in time specific customers are likely to be in risk.
# MAGIC   
# MAGIC   * `Hardware Failures:` The quality of experience a customer has with your products and services plays a key role in the decision to renew or cancel. The network itself is at the epicenter of this experience. With time to failure as the event of interest, Survival Analysis can be used to predict when hardware will need to be repaired or replaced.
# MAGIC   
# MAGIC   * `Device and Data Plan Upgrades:` There are key moments in a customer's lifecycle when changes to their plan take place. With the event of interest being a plan change, Survival Analysis can be used to predict when such a change will take place and then actions can be taken to positively influence the selected products or services.

# COMMAND ----------

# MAGIC %md
# MAGIC ## About This Series of Notebooks
# MAGIC 
# MAGIC * The intent for this series of notebooks is to help fast track the use of Survival Analysis within your line of business by equipping you with the information needed to do so.
# MAGIC 
# MAGIC * In support of this goal, we will apply and review several techniques that are commonly used for Survival Analysis:
# MAGIC   * Kaplan-Meier & the Log-Rank Test
# MAGIC   * Cox Proportional Hazards
# MAGIC   * Accelerated Failure Time
# MAGIC   
# MAGIC * Lastly, we will use the output of our Survival Analysis model as an input for a Customer Lifetime Value dashboard.

# COMMAND ----------

# MAGIC %md
# MAGIC ## About the Data
# MAGIC 
# MAGIC * The dataset used in this series of notebooks comes from [IBM](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv) and is meant to resemble a fictitious Telecommunications company. Each record in this dataset represents a subscriber and contains information about their respective demographics, service plan, media usage, and subscription status. Most important, as we'll see in a bit, this dataset contains two columns that are required for this form of analysis:
# MAGIC   * `Tenure:` the duration that a customer has been with the company (if still a subscriber) or was with the company prior to churning.
# MAGIC   * `Churn:` a Boolean indicating whether the customer is still a subscriber or not.
# MAGIC   
# MAGIC * Before we get started with the analysis, let's:
# MAGIC    * Download [IBM's Telco dataset](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv).
# MAGIC    * Use Delta Lake to store the data in Bronze and Silver tables.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure the Environment
# MAGIC 
# MAGIC * Import libraries
# MAGIC 
# MAGIC * Set configuration
# MAGIC   * `database_name:` the database name given to the database that will hold bronze-level and silver-level tables
# MAGIC   * `driver_to_dbfs_path:` the directory used to persist IBM's Telco dataset as-is
# MAGIC   * `bronze_tbl_path:` the directory used to persist IBM's Telco dataset in Delta format
# MAGIC   * `silver_tbl_path:` the directory used to persist a curated version of IBM's Telco dataset
# MAGIC   * `bronze_tbl_name:` the table name given for the bronze-level data
# MAGIC   * `silver_tbl_name:` the table name given for the silver-level data

# COMMAND ----------

# Load libraries
import shutil
from pyspark.sql.functions import col, when
from pyspark.sql.types import StructType,StructField,DoubleType, StringType, IntegerType

# COMMAND ----------

# Set config for database name, file paths, and table names
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get().split('@')[0]
username_sql = user.replace(".", "_")
database_name = f'dm_ibm_telco_churn_{username_sql}'
data_path = '/home/{}/ibm-telco-churn'.format(user) 
driver_to_dbfs_path = '{}/Telco-Customer-Churn.csv'.format(data_path) 
bronze_tbl_path = '{}/bronze/'.format(data_path) 
silver_tbl_path = '{}/silver/'.format(data_path) 

bronze_tbl_name = 'bronze_customers'
silver_tbl_name = 'silver_monthly_customers'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download and Load the Data
# MAGIC 
# MAGIC * Data Source:
# MAGIC   * Dataset title: Telco Customer Churn
# MAGIC   * Dataset source URL: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
# MAGIC   * Dataset source description: Github repository managed by IBM
# MAGIC   * Dataset license: please see dataset source URL above
# MAGIC 
# MAGIC * Since this is a relatively small dataset, we download it directly to the driver of our Spark Cluster. We then move it to the Databricks Filesystem (DBFS).
# MAGIC 
# MAGIC * Before reading in the data, we first define the schema. This is good practice when it's possible to do as it avoids scanning files to infer the schema and affords you control over data types. 

# COMMAND ----------

# MAGIC %sh
# MAGIC cd /databricks/driver 
# MAGIC wget https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

# COMMAND ----------

# Define schema
schema = StructType([
  StructField('customerID', StringType()),
  StructField('gender', StringType()),
  StructField('seniorCitizen', DoubleType()),
  StructField('partner', StringType()),
  StructField('dependents', StringType()),
  StructField('tenure', DoubleType()),
  StructField('phoneService', StringType()),
  StructField('multipleLines', StringType()),
  StructField('internetService', StringType()), 
  StructField('onlineSecurity', StringType()),
  StructField('onlineBackup', StringType()),
  StructField('deviceProtection', StringType()),
  StructField('techSupport', StringType()),
  StructField('streamingTV', StringType()),
  StructField('streamingMovies', StringType()),
  StructField('contract', StringType()),
  StructField('paperlessBilling', StringType()),
  StructField('paymentMethod', StringType()),
  StructField('monthlyCharges', DoubleType()),
  StructField('totalCharges', DoubleType()),
  StructField('churnString', StringType())
  ])

# COMMAND ----------

# Load data
dbutils.fs.cp('file:/databricks/driver/Telco-Customer-Churn.csv', driver_to_dbfs_path)
bronze_df = spark.read.format('csv').schema(schema).option('header','true')\
               .load(driver_to_dbfs_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Curate Data for Silver Table
# MAGIC 
# MAGIC * In an effort to keep our analysis focused, we will apply a few transformations to the original dataset.
# MAGIC   * Transform churn column to Boolean
# MAGIC   * Filter to internet subscribers with a month-to-month contract
# MAGIC 
# MAGIC * We refer to this curated dataset as the silver table

# COMMAND ----------

# Construct silver table
silver_df = bronze_df.withColumn('churn',when(col('churnString') == 'Yes',1).when(col('churnString') == 'No',0).otherwise('Unknown'))\
                     .drop('churnString').filter(col('contract') == 'Month-to-month')\
                     .filter(col('internetService') != 'No')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Database

# COMMAND ----------

# Delete the old database and tables if needed
_ = spark.sql('DROP DATABASE IF EXISTS {} CASCADE'.format(database_name))

# Create database to house tables
_ = spark.sql('CREATE DATABASE {}'.format(database_name))
_ = spark.sql('USE {}'.format(database_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Data to Delta Lake

# COMMAND ----------

# Drop any old delta lake files if needed (e.g. re-running this notebook with the same bronze_tbl_path and silver_tbl_path)
shutil.rmtree('/dbfs'+bronze_tbl_path, ignore_errors=True)
shutil.rmtree('/dbfs'+silver_tbl_path, ignore_errors=True)

# COMMAND ----------

# Write out bronze-level data to Delta Lake
_ = bronze_df.write.format('delta').mode('overwrite').save(bronze_tbl_path)

# COMMAND ----------

# Write out silver-level data to Delta lake
_ = silver_df.write.format('delta').mode('overwrite').save(silver_tbl_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Tables
# MAGIC 
# MAGIC * These tables will point to the data that you recently wrote out to Delta Lake

# COMMAND ----------

# Create bronze table
_ = spark.sql('''
  CREATE TABLE `{}`.{}
  USING DELTA 
  LOCATION '{}'
  '''.format(database_name,bronze_tbl_name,bronze_tbl_path))

# COMMAND ----------

# Create silver table
_ = spark.sql('''
  CREATE TABLE `{}`.{}
  USING DELTA 
  LOCATION '{}'
  '''.format(database_name,silver_tbl_name,silver_tbl_path))


# COMMAND ----------

# MAGIC %md
# MAGIC ## View Tables

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from bronze_customers

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from silver_monthly_customers

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC * Now that our data is in place, we can proceed to analysis, starting with Kaplan-Meier

# COMMAND ----------

# MAGIC %md
# MAGIC Copyright Databricks, Inc. [2021]. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC |Library Name|Library license | Library License URL | Library Source URL |
# MAGIC |---|---|---|---|
# MAGIC |Lifelines|MIT License|https://github.com/CamDavidsonPilon/lifelines/blob/master/LICENSE|https://github.com/CamDavidsonPilon/lifelines|
# MAGIC |Matplotlib|Python Software Foundation (PSF) License |https://matplotlib.org/stable/users/license.html|https://github.com/matplotlib/matplotlib|
# MAGIC |Numpy|BSD-3-Clause License|https://github.com/numpy/numpy/blob/master/LICENSE.txt|https://github.com/numpy/numpy|
# MAGIC |Pandas|BSD 3-Clause License|https://github.com/pandas-dev/pandas/blob/master/LICENSE|https://github.com/pandas-dev/pandas|
# MAGIC |Python|Python Software Foundation (PSF) |https://github.com/python/cpython/blob/master/LICENSE|https://github.com/python/cpython|
# MAGIC |Seaborn|BSD-3-Clause License|https://github.com/mwaskom/seaborn/blob/master/LICENSE|https://github.com/mwaskom/seaborn|
# MAGIC |Spark|Apache-2.0 License |https://github.com/apache/spark/blob/master/LICENSE|https://github.com/apache/spark|
