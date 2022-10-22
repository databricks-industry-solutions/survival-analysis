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
# MAGIC ## Accelerated Failure Time
# MAGIC 
# MAGIC ### In this lesson you:
# MAGIC * Fit a Log-Logistic Accelerated Failure Time model to IBM's Telco dataset
# MAGIC * Interpret the statistical output of the Accelerated Failure Time model
# MAGIC * Determine whether the model adheres to or violates the underlying assumptions 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Overview
# MAGIC 
# MAGIC * The canonical example of Accelerated Failure Time models, shared by Kleinbaum & Klein in [Survival Analysis: A Self-Learning Text](https://www.springer.com/gp/book/9781441966452), is the lifespan of dogs. It is commonly accepted that dogs age 7x faster than humans. They go through the same lifestages that we do, just faster.
# MAGIC 
# MAGIC * In contrast to Kaplan-Meier and Cox Proportional Hazards, Accelerated Failure Time is a parametric model. This means that the outcome variable is assumed to follow a specified distribution. Parametric models are typically less 'flexible' than non-parametric and semi-parametric models but can be a good choice when you're comfortable with specifying the distribution of the outcome variable.
# MAGIC 
# MAGIC * Similar to Cox Proportional Hazards, the Accelerated Failure Time model has underlying assumptions to be aware of. These assumptions are covered below.

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Accelerated Failure Time Model Equation
# MAGIC 
# MAGIC * Using the Accelerated Failure Time equation below, if we were to define group A as humans and group B as dogs, then the acceleration factor would be 7.  Similarly, if we define group A as dogs and group B as humans, then the acceleration factor would be 1/7.
# MAGIC 
# MAGIC * The specification for lambda, which represents the accelerated failure rate, is intentionally generalized here. In practice, the survival function for the accelerated failure rate includes one or more parameters. For example, the specification when using log-logistic accelerated failure time is: 1/(1+lambda x t ^ p).
# MAGIC 
# MAGIC * The full specification of the accelerated failure rate is most relevant when using log-log plots to verify whether the accelerated failure time assumptions have been violated. This is covered in further detail below.
# MAGIC 
# MAGIC <div >
# MAGIC   <img src ='https://brysmiwasb.blob.core.windows.net/demos/images/ME_AFT.png' width=600>
# MAGIC </div>

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

# Load the telco silver table
telco_pd = spark.table('silver_monthly_customers').toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## One-Hot Encode the Categorical Variables
# MAGIC 
# MAGIC * As is the case when using the Lifelines library to fit a Cox Proportional Hazards model, you must one-hot encode your categorical columns when using an Accelerated Failure Time model. We have hand-selected 8 variables to use for this notebook.
# MAGIC 
# MAGIC * After one-hot encoding the variables, we then created a new dataframe consisting of only the columns we need in order to fit the model.
# MAGIC 
# MAGIC * When creating the dataframe that you will use to fit the model, it is important that you drop one column for each one-hot encoded variable.  For example, we included dependents_yes and dropped dependents_no.  If you neglect to do this, you will likely receive an error when fitting the model that states that there are multi-collinearity issues with your data. This means that certain columns are highly correlated. For example, if you know that dependents_yes = 1 then you in turn know that dependents_no = 0.
# MAGIC 
# MAGIC * You can take one of two approaches when dropping variables. When using Pandas' get_dummies() function, you can simply set drop_first = True. An alternative approach, which is the one taken here, is to hand-select which variables to drop. In this notebook, we dropped the value that had a Kaplan-Meier survival probability curve most similar to the population.  While this is by no means necessary, it affords an intuitive interpretation as the baseline closely resembles the population.

# COMMAND ----------

encode_cols = ['partner','multipleLines','internetService','onlineSecurity', 'onlineBackup','deviceProtection','techSupport','paymentMethod']

encoded_pd = pd.get_dummies(telco_pd,
               columns=encode_cols,
               prefix=encode_cols,
               drop_first=False)

encoded_pd.head()

# COMMAND ----------

survival_pd = encoded_pd[['churn','tenure','partner_Yes', 'multipleLines_Yes', \
                          'internetService_DSL','onlineSecurity_Yes','onlineBackup_Yes','deviceProtection_Yes','techSupport_Yes',\
                          'paymentMethod_Bank transfer (automatic)','paymentMethod_Credit card (automatic)' ]]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit the Accelerated Failure Time Model
# MAGIC 
# MAGIC Similar to Kaplan-Meier and Cox Proportional Hazards, the first step when using the Lifelines library for Accelerated Failure Time is to fit the model. 
# MAGIC * In this notebook, the type of distribution that we have specified for the outcome variable is log-logistic. This is reflected by the use of `LogLogisticAFTFitter` when fitting the model. Other options that are available through Lifelines include: `WeibullAFTFitter`, `LogNormalAFTFitter`. After we fit the model and assess the results, we will assess whether `LogLogistic` is the appropriate type of distribution to specify for this dataset. 
# MAGIC 
# MAGIC * We provided three parameters to the fitted model:
# MAGIC   * `suvival_pd`: the dataframe used to fit the model
# MAGIC   * `Tenure:` the duration that a customer has been with the company (if still a subscriber) or was with the company prior to churning.
# MAGIC   * `Churn:` a Boolean indicating whether the customer is still a subscriber or not.

# COMMAND ----------

aft = LogLogisticAFTFitter()
aft.fit(survival_pd, duration_col='tenure', event_col='churn')

# COMMAND ----------

# Note: the output is exponentiated because it is initially in log-odds
print("Median Survival Time:{:.2f}".format(np.exp(aft.median_survival_time_)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Assess the Results of the Fitted Model
# MAGIC 
# MAGIC There are three key questions to ask when initially assessing the model:
# MAGIC 1. **Is each covariate statistically significant?**
# MAGIC   * In the column labeled `p` below, it can be seen that the p-value for each column is below < 0.005. Therefore, each of the columns is statistically significant and safe to include. 
# MAGIC   * Similar to other forms of regression, in the event that a variable is not statisitically significant, you can drop that column from your analysis or recategorize the corresponding values.
# MAGIC 
# MAGIC 2. **How confident are we in the coefficient estimates?**
# MAGIC   * Upper and lower bounds for each coefficient and exp(coefficient) are provided in the model summary below (e.g. `exp(coef) lower 95%`, `exp(coef) upper 95%`).
# MAGIC   * Shown below the model summary, these bounds can also be viewed visually. Although it's not seen here, note that when viewing the bounds in this way, a variable can be deemed as not signficant when it's corresponding box-and-whisker plot crosses the value 1.
# MAGIC 
# MAGIC 3. **What is the effect of each covariate on the hazard ratio?**
# MAGIC   * Using internetService_DSL as an example, it's shown below that `coef = 0.38` and `exp(coef) = 1.47`. Referring back to the Accelerated Failure Time equation, this means that a customer's time-until-churn is accelerated by a factor of 1.47 when they have Fiber Optic as their internet service.  Note that Fiber Optic is the baseline value and corresponds to Group A in the equation shared above.
# MAGIC   
# MAGIC Note that this output also includes several metrics that can be used when comparing models:
# MAGIC * `Concordance`
# MAGIC * `AIC`
# MAGIC * `Log-likelihood Ratio`

# COMMAND ----------

aft.print_summary()

# COMMAND ----------

aft.plot()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify if the Model Adheres to Assumptions
# MAGIC 
# MAGIC * As we saw with Cox Proportional Hazards, log-log plots are very helpful for assessing whether a model violates assumptions. 
# MAGIC   
# MAGIC   * For Cox Proportional Hazards, log-log plots were created with log(time) on the x-axis and the log-log of the survival function on the y-axis. 
# MAGIC   
# MAGIC   * For Accelerated Failure Time, the formula that is used for the y-axis is dependent upon the type of distribution specified for the outcome variable. Since we are using log-logistic here, the formula for the y-axis is: log(1 - survival_function) / survival_function.
# MAGIC 
# MAGIC * There are two underlying assumptions to assess when using an Accelerated Failure Time model:
# MAGIC   
# MAGIC   * Does the model adhere to the `Proportional Odds` assumption? - the answer is yes when lines in the plot are parallel.
# MAGIC   
# MAGIC   * Is the `specified distribution` appropriate for this model? - the answer is yes when the lines are straight.
# MAGIC 
# MAGIC * How does our model fare?
# MAGIC   
# MAGIC   * For the most part, the lines in each of the plots are relatively straight. There is some deviation but not bad overall. This implies that selecting log-logistic as the specified distribution is a reasonable choice.
# MAGIC   
# MAGIC   * For the most part, the lines in each of the plots are not parallel.  This implies that Accelerated Failure Time is not appropriate for the specified model.

# COMMAND ----------

# Fit the Kaplan-Meier model
from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()

T=telco_pd['tenure'] #duration
C=telco_pd['churn'].astype(float) #event observed

kmf.fit(T,C)

# COMMAND ----------

# Utility function for plotting
import matplotlib.pyplot as plt

def plot_km_logOdds(col):
  ax = plt.subplot(111)
  for r in telco_pd[col].unique():
    ix = telco_pd[col] == r
    kmf.fit(T[ix], C[ix],label=r)
    sf = kmf.survival_function_
    sf['failureOdds'] = (np.log(1-sf))/sf
    sf['logTime'] = np.log(sf.index)
    plt.plot(sf['logTime'],sf['failureOdds'])

# COMMAND ----------

plot_km_logOdds('partner')

# COMMAND ----------

plot_km_logOdds('multipleLines')

# COMMAND ----------

plot_km_logOdds('internetService')

# COMMAND ----------

plot_km_logOdds('onlineSecurity')

# COMMAND ----------

plot_km_logOdds('onlineBackup')

# COMMAND ----------

plot_km_logOdds('deviceProtection')

# COMMAND ----------

plot_km_logOdds('techSupport')

# COMMAND ----------

plot_km_logOdds('paymentMethod')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Closing Thoughts
# MAGIC 
# MAGIC * Cox Proportional Hazards is by far the most popular method used for Time-to-Event (or Survival) Analysis. A big part of why this is the case is that as a semi-parametric model, you are not required to specify the form of your outcome variable. Compounding this benefit, this model tends to perform quite well, and as discussed in the previous notebook, there are a number of techniques available for further improvement.
# MAGIC 
# MAGIC * While Cox Proportional Hazards is often a good choice, it's beneficial to know that other options do exist. In this notebook, we reviewed a fully parametric model referred to as Accelerated Failure Time.
# MAGIC 
# MAGIC * In practice, if your goal is inference, you should select the model that best adheres to the underlying assumptions.  If you goal is prediction, you should select the model that performs best with respect to your chosen evaluation metric. For this specific dataset, it would be advised to further explore options 2, 3, and 4 shared in the `closing thoughts` section of the `Cox Proportional Hazards` notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC * In the next notebook, we will demonstrate how you can use the output of a Survival Analysis model as an input for calculating customer lifetime value.

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
