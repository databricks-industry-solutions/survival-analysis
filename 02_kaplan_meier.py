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
# MAGIC 
# MAGIC ## Kaplan-Meier
# MAGIC ### In this lesson you:
# MAGIC * Fit Kaplan-Meier survival probability curves to IBM's Telco dataset.
# MAGIC * Visually assess survival probability curves at the population-level and the covariate level.
# MAGIC * Use the log-rank test to determine if survival curves are statistically equivalent.
# MAGIC * Extract survival probabilities for subsequent modeling.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Overview
# MAGIC 
# MAGIC * Kaplan-Meier is a statistical method that is used to construct survival probablity curves. This method takes censoring into account, therefore overcoming the issue of underestimating survival probabilities that ocurrs when using mean or median.
# MAGIC * The log-rank test is a chi-square test that is used to test the null hypothesis that two or more survival curves are statistically equivalant.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure the Environment
# MAGIC 
# MAGIC * Install the Lifelines library using PyPi
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
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from lifelines.statistics import pairwise_logrank_test

# COMMAND ----------

# Load the telco silver table
telco_pd = spark.table('silver_monthly_customers').toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit the Kaplan-Meier Model
# MAGIC 
# MAGIC The first step when using Lifelines for Kaplan-Meier is to fit the model. This step requires two parameters: tenure and churn.
# MAGIC * `Tenure:` the duration that a customer has been with the company (if still a subscriber) or was with the company prior to churning.
# MAGIC * `Churn:` a Boolean indicating whether the customer is still a subscriber or not.

# COMMAND ----------

kmf = KaplanMeierFitter()

T=telco_pd['tenure']
C=telco_pd['churn'].astype(float)

kmf.fit(T,C)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visually Assess the Population-Level Survival Curve
# MAGIC * `The Plot:` As seen in the plot below, the survival probability curve is plotted with time on the x-axis and survival probability on the y-axis.
# MAGIC 
# MAGIC * `Interpretation:` In the purest sense, the probability of a customer surviving **at least** 0 months is 100%. This is represented by the point (0,1.0) in the plot below. Moving down the survival curve to the median (34 months), it can be said that a customer has a 50% probability of surviving **at least** 34 months, given that they have survived 33 months. Please note that the last clause 'given that...' signifies that this is a conditional probability.
# MAGIC 
# MAGIC * `Confidence Intervals:` The light blue border surrounding the survival probability curve represents the confidence interval. Thie wider the interval, the lower the confidence. As seen in the plot below, confidence in the estimates decrease as the timeline increases. While this reduced confidence is likely due to having less data, it is also intuitive that we would have more confidence in our predictions for the near-term than in our predictions for the longer-term.

# COMMAND ----------

kmf.plot(title='Kaplan-Meier Survival Curve: Population level')

# COMMAND ----------

kmf.median_survival_time_

# COMMAND ----------

# MAGIC %md
# MAGIC ## Assess Survival Probabilities at the Covariate-Level
# MAGIC 
# MAGIC * When viewing Kaplan-Meier curves at the covariate-level, it's ideal to see some level of divergence between the groups as this indicates a difference that can be used for predictions. `Online Security` is a good example of this.
# MAGIC 
# MAGIC * For the purposes of prediction, it's less ideal to see the survival curves very close together, as is the case for the variable `Gender`. When survival curves are very close together, you may want to check whether they are statistically equivalent. This is the purpose of the log-rank test. The null hypothesis for the log-rank states that the groups are statistically equivalent.
# MAGIC 
# MAGIC * As seen below, the p-value in our log-rank test for `Gender` is greater than 0.05 and therefore, we cannot reject the null hypothesis that the two groups are statistically equivalent.
# MAGIC 
# MAGIC * Although it's not useful for prediction to find out that the two groups are statistically equivalent, there are benefits from an inference standpoint. For example, suppose you have a new promotion that provides subscribers with free access to a streaming service. If you find out that those with the service are the same as those without the service, you may ask whether the return on providing free access is high enough to cover costs. In contrast, suppose you want to understand the effect of throttling. If you find that the effect is minimal, you may have cause to continue doing so.

# COMMAND ----------

# Helper function for plotting Kaplan-Meier curves at the covariate level
def plot_km(col):
  ax = plt.subplot(111)
  for r in telco_pd[col].unique():
    ix = telco_pd[col] == r
    kmf.fit(T[ix], C[ix],label=r)
    kmf.plot(ax=ax)
    
# Helper function for printing out Log-rank test results
def print_logrank(col):
  log_rank = pairwise_logrank_test(telco_pd['tenure'], telco_pd[col], telco_pd['churn'])
  return log_rank.summary

# COMMAND ----------

plot_km('gender')

# COMMAND ----------

print_logrank('gender')

# COMMAND ----------

plot_km('onlineSecurity')

# COMMAND ----------

print_logrank('onlineSecurity')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Assess Survival Probabilities at the Covariate-Level (continued)
# MAGIC 
# MAGIC * For convenience, Kaplan-Meier curves and log-rank tests are provided for every variable in the dataset below.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Gender

# COMMAND ----------

plot_km('gender')

# COMMAND ----------

print_logrank('gender')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Senior Citizen

# COMMAND ----------

plot_km('seniorCitizen')

# COMMAND ----------

print_logrank('seniorCitizen')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Partner

# COMMAND ----------

plot_km('partner')

# COMMAND ----------

print_logrank('partner')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Dependents

# COMMAND ----------

plot_km('dependents')

# COMMAND ----------

print_logrank('dependents')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Phone Service

# COMMAND ----------

plot_km('phoneService')

# COMMAND ----------

print_logrank('phoneService')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Multiple Lines

# COMMAND ----------

plot_km('multipleLines')

# COMMAND ----------

print_logrank('multipleLines')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Internet Service

# COMMAND ----------

plot_km('internetService')

# COMMAND ----------

print_logrank('internetService')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Streaming TV

# COMMAND ----------

plot_km('streamingTV')

# COMMAND ----------

print_logrank('streamingTV')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Streaming Movies

# COMMAND ----------

plot_km('streamingMovies')

# COMMAND ----------

print_logrank('streamingMovies')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Online Security

# COMMAND ----------

plot_km('onlineSecurity')

# COMMAND ----------

print_logrank('onlineSecurity')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Online Backup

# COMMAND ----------

plot_km('onlineBackup')

# COMMAND ----------

print_logrank('onlineBackup')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Device Protection

# COMMAND ----------

plot_km('deviceProtection')

# COMMAND ----------

print_logrank('deviceProtection')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Tech Support

# COMMAND ----------

plot_km('techSupport')

# COMMAND ----------

print_logrank('techSupport')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Paperless Billing

# COMMAND ----------

plot_km('paperlessBilling')

# COMMAND ----------

print_logrank('paperlessBilling')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Payment Method

# COMMAND ----------

plot_km('paymentMethod')

# COMMAND ----------

print_logrank('paymentMethod')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract Survival Probabilities
# MAGIC 
# MAGIC * After you complete your analysis, you may want to extract the survival probabilities to use for another application. Lifelines makes this quite easy to do.
# MAGIC 
# MAGIC * Please note that later on, we will extract the output from a Cox Proportional Hazard model and use it as an input for a Customer Lifetime Value dashboard.

# COMMAND ----------

def get_survival_probs(col,val):
  ix = telco_pd[col] == val
  return kmf.fit(T[ix],C[ix],label=val)  

# COMMAND ----------

# Extract survival probabilities for customers with internetService = DSL
sp_internet_dsl = get_survival_probs('internetService','DSL')

# COMMAND ----------

pd.DataFrame(sp_internet_dsl.survival_function_at_times(range(0,10)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Closing Thoughts
# MAGIC 
# MAGIC * Kaplan-Meier is extremely useful for univariate analysis as it helps establish more intuition for the data at hand. Cox Proportional Hazards, which we cover in the next notebook, is more useful for multi-variate analysis.
# MAGIC 
# MAGIC * In the plots above, you may notice that the survival curves for a given covariate crossover each other. This is a red flag that we will discuss in the next notebook.
# MAGIC 
# MAGIC * As you'll soon see, Kaplan-Meier curves have some interesting applications that make them useful even when leveraging different techniques.

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
