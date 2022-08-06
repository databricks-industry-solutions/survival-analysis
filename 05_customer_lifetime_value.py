# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left">
# MAGIC   <img src="https://brysmiwasb.blob.core.windows.net/demos/images/ME_solution-accelerator.png"; width="50%">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Customer Lifetime Value
# MAGIC 
# MAGIC ### In this lesson you:
# MAGIC * Fit a Cox Proportional Hazard model to IBM's Telco dataset.
# MAGIC * Interpret the statistical output of the Cox Proportional Hazard Model.
# MAGIC * Determine whether the model adheres to or violates the proportional hazard assumption. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure the Environment
# MAGIC 
# MAGIC * Install the lifelines library using PyPi
# MAGIC * Import relevant libraries
# MAGIC * Load the telco silver table constructed in `01 Intro`
# MAGIC   * Dataset title: Telco Customer Churn
# MAGIC   * Dataset source URL: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
# MAGIC   * Dataset source description: Github repository managed by IBM
# MAGIC   * Dataset license: please see dataset source URL above

# COMMAND ----------

# MAGIC %run ./config/notebook_config

# COMMAND ----------

# Import libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from lifelines import WeibullAFTFitter,LogNormalAFTFitter,LogLogisticAFTFitter
from lifelines.fitters.coxph_fitter import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

# COMMAND ----------

# Load the Telco silver table
telco_pd = spark.table('silver_monthly_customers').toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit the Cox Proportional Hazards Model

# COMMAND ----------

# Encode columns of interest
encode_cols = ['dependents','internetService','onlineBackup','techSupport','paperlessBilling']

encoded_pd = pd.get_dummies(telco_pd,
               columns=encode_cols,
               prefix=encode_cols,
               drop_first=False)

encoded_pd.head()


# COMMAND ----------

# Create new dataframe consisting of only the variables needed to fit the model
survival_pd = encoded_pd[['churn','tenure','dependents_Yes','internetService_DSL','onlineBackup_Yes','techSupport_Yes']]

# COMMAND ----------

# Cast churn column as a float
survival_pd.loc[:,'churn'] = survival_pd.loc[:,'churn'].astype('float')

# COMMAND ----------

cph = CoxPHFitter(alpha=0.05)
cph.fit(survival_pd, 'tenure', 'churn')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Widgets for Dashboard

# COMMAND ----------

# Create widgets
cols = ['dependents_Yes','internetService_DSL','onlineBackup_Yes','techSupport_Yes','partner_Yes',' internal rate of return']

dbutils.widgets.removeAll()

for col in cols:
  if col == ' internal rate of return':
    dbutils.widgets.text(' internal rate of return',defaultValue='0.10')
  else:
    dbutils.widgets.dropdown(col,choices=['0','1'],defaultValue='0')

# COMMAND ----------

# Construct dataframe with values from widgets
def get_widget_values():
  widget_dict = {col:dbutils.widgets.get(col) for col in cols}
  return pd.DataFrame.from_dict(widget_dict,orient='index').T

def get_payback_df():
  df = get_widget_values()
  irr = df[' internal rate of return'].astype('float64')[0]/12
  cohort_df = pd.concat([pd.DataFrame([1.00]),round(cph.predict_survival_function(df),2)]).rename(columns={0:'Survival Probability'})
  cohort_df['Contract Month'] = cohort_df.index.astype('int')
  cohort_df['Monthly Profit for the Selected Plan'] = 30
  cohort_df['Avg Expected Monthly Profit'] = round(cohort_df['Survival Probability'] * cohort_df['Monthly Profit for the Selected Plan'],2)
  cohort_df['NPV of Avg Expected Monthly Profit'] = round(cohort_df['Avg Expected Monthly Profit'] / ((1+irr)**cohort_df['Contract Month']),2)
  cohort_df['Cumulative NPV'] = cohort_df['NPV of Avg Expected Monthly Profit'].cumsum()
  cohort_df['Contract Month'] = cohort_df['Contract Month'] + 1
  return cohort_df[['Contract Month','Survival Probability','Monthly Profit for the Selected Plan','Avg Expected Monthly Profit','NPV of Avg Expected Monthly Profit','Cumulative NPV']].set_index('Contract Month')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Table for Dashboard
# MAGIC 
# MAGIC * This table is generated by calling `get_payback_df()`, which uses values from the widgets as parameters.
# MAGIC   
# MAGIC   * `Survival Probability:` Probablities are extracted from the model using `predict_survival_function()`
# MAGIC   
# MAGIC   * `Monthly Profit for the Selected Plan:` This is currently hard-coded to 30 for illustraton purposes.  In practice, you would use the respective internal data.
# MAGIC   
# MAGIC   * `Avg Expected Monthly Profit:` On average, the expected monthly profit for a given customer is `Survival Probability` x `Monthly Profit` for the respective plan.
# MAGIC   
# MAGIC   * `NPV of Avg Expected Monthly Profit:` Since a dollar received today is worth more than a dollar received in a few years for now, it is common to use `Net Present Value`. The formula used here is:
# MAGIC     * `Avg Expected Monthly Profit` / ((1 + `Internal Rate of Return`) ^ `Contract Month`)
# MAGIC     
# MAGIC     * The default value of `Internal Rate of Return` is set to 10%, however, you should use the number that is accepted in your line of business. 
# MAGIC   
# MAGIC   * `Cumulative NPV:` Cumulative sum of `NPV of Avg Expected Monthly Profit`

# COMMAND ----------

pd.options.display.max_rows = 25
get_payback_df()[0:25]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Cumulative NPV Chart for Dashboard
# MAGIC 
# MAGIC * Many organizations focus on the `payback period` as a means for optimizing spend. This chart illustrates that maximum amount you would be willing to spend to acquire a customer (assuming that you are not willing to lose money on that cohort).

# COMMAND ----------

import seaborn as sns
ax = sns.barplot(['12 Months','24 Months','36 Months'],get_payback_df().iloc[[11,23,35],:]['Cumulative NPV'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Survival Probability Curve Chart for Dashboard
# MAGIC 
# MAGIC * This is the same type of visual seen in the previous notebooks but customized to the parameters entered into the widgets.

# COMMAND ----------

# DBTITLE 1,Survival Probability Curve
sns.lineplot(get_payback_df().index,get_payback_df()['Survival Probability'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### View Dashboard
# MAGIC * To view the dashboard created in this notebook:
# MAGIC   * Click on `View: Standard` in the top nav
# MAGIC   * Click on `CAC vs. LTV` in the dropdown menu

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
