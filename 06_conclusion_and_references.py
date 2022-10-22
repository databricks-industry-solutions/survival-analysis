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
# MAGIC ## Recap
# MAGIC 
# MAGIC * The intent for this series of notebooks is to help fast track the use of Survival Analysis within your line of business by equipping you with the information needed to do so.
# MAGIC 
# MAGIC * In support of this goal, we applied and reviewed several techniques that are commonly used for Survival Analysis:
# MAGIC   * Kaplan-Meier & the Log-Rank Test
# MAGIC   * Cox Proportional Hazards
# MAGIC   * Accelerated Failure Time
# MAGIC   
# MAGIC * Lastly, we will used the output of our Survival Analysis model as an input for a Customer Lifetime Value dashboard.

# COMMAND ----------

# MAGIC %md
# MAGIC ## References
# MAGIC 
# MAGIC The following resources were referred to during the creation of this solution accelerator and are highly recommended.
# MAGIC * [Kleinbaum, D., & Klein, M. (2012). _Survival analysis: A self-learning text_ (3rd ed.). Springer.](https://www.springer.com/gp/book/9781441966452)
# MAGIC * [Lifelines documentation](https://lifelines.readthedocs.io/en/latest/)

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
