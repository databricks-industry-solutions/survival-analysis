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
# MAGIC ## Cox Proportional Hazards
# MAGIC 
# MAGIC ### In this lesson you:
# MAGIC * Fit a Cox Proportional Hazard model to IBM's Telco dataset.
# MAGIC * Interpret the statistical output of the Cox Proportional Hazard Model.
# MAGIC * Determine whether the model adheres to or violates the proportional hazard assumption. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Overview
# MAGIC 
# MAGIC * In contrast to Kaplan-Meier, Cox Proportional Hazards can be used for multi-variate analysis.
# MAGIC 
# MAGIC * Similar to Kaplan-Meier, Cox Proportional Hazards can be used to plot survival probability curves but the way it is done differs mathematically.  The result is referred to as adjusted survival probability curves because you adjust for other covariates.

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Cox Proportional Hazards Equation
# MAGIC 
# MAGIC * Whereas Kaplan-Meier is used to estimate the probability of survival, Cox Proportional Hazards is used to estimate the hazard ratio.  The hazard ratio represents the difference in hazard that exists between two individuals (or groups). The hazard is essentially the inverse of survival, or the probability of failure.  The fact that Kaplan-Meier estimates the probability of survival whereas Cox Proportional Hazards estimates a hazard ratio is not a concern because as long as you have one, you can calculate the other. The Lifelines library makes this easy to do.
# MAGIC 
# MAGIC * The Cox Proportional Hazards equation states that the hazard ratio is the product of two terms: the baseline hazard and the partial hazard.
# MAGIC 
# MAGIC * The baseline hazard is simply a baseline. It's the hazard that exists when each variable is set to a specific value. For example:
# MAGIC 
# MAGIC |Variable|Value|
# MAGIC |---|---|
# MAGIC |gender|Female|
# MAGIC |seniorCitizen|No|
# MAGIC |partner|No|
# MAGIC |dependents|No|
# MAGIC |phoneService|Yes|
# MAGIC   
# MAGIC * The partial hazard represents the change in the hazard that occurs when the value for a variable is different from the baseline. At any given time, zero or more variables can contain a value different from the baseline. As seen in the equation below, the resulting change in the hazard is a linear combination of the parameters / variables.
# MAGIC 
# MAGIC * If every variable were set to its corresponding baseline value, then the partial hazard would equal 1 (since e^0 = 1) and the hazard ratio would equal the baseline hazard.
# MAGIC 
# MAGIC <div >
# MAGIC   <img src ='https://brysmiwasb.blob.core.windows.net/demos/images/ME_CPH.png' width=600>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Proportional Hazards Assumption
# MAGIC 
# MAGIC * A subtle but critical element of the Cox Proportional Hazard equation is that the baseline hazard is a function of time _t_, but not the parameters, whereas the partial hazard is a function of the parameters, but not time. This underpins what is referred to as the proportional hazard assumption.
# MAGIC 
# MAGIC * The proportional hazard assumption states that in the context of a Cox Proportional Hazard model, the hazard ratio between two groups is proportional over time. This assumption is implicit in the equation above because the lack of _t_ in the partial hazard means that the partial hazard changes the hazard ratio by some factor, independent of time. 

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8,3))

ax1.plot([1,2,3,4,5],[1,2,3,4,5])
ax1.plot([1,2,3,4,5],[2,3,4,5,6])
ax1.title.set_text('Proportional')
ax1.axes.get_xaxis().set_visible(False)
ax1.axes.get_yaxis().set_visible(False)

ax2.plot([1,2,3,4,5],[1,2,3,4,5])
ax2.plot([1,2,3,4,5],[2,4,6,8,10])
ax2.title.set_text('Not Proportional')
ax2.axes.get_xaxis().set_visible(False)
ax2.axes.get_yaxis().set_visible(False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure the Environment
# MAGIC 
# MAGIC * Install the Lifelines library using PyPi
# MAGIC * Import relevant libraries
# MAGIC * Load the telco silver table constructed in the notebook `01 Intro`
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

from lifelines.fitters.coxph_fitter import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from lifelines import KaplanMeierFitter

# COMMAND ----------

# Load the telco silver table
telco_pd = spark.table('silver_monthly_customers').toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## One-Hot Encode the Categorical Variables
# MAGIC 
# MAGIC * In order to fit a Cox Proportional Hazards model using the Lifelines library, you must first one-hot encode your categorical columns. We have hand-selected 5 variables to use for this notebook.
# MAGIC 
# MAGIC * After one-hot encoding the variables, we then created a new dataframe consisting of only the columns we need to fit the model.
# MAGIC 
# MAGIC * When creating the dataframe that you will use to fit the model, it is important that you drop one column for each one-hot encoded variable.  For example, we included dependents_yes and dropped dependents_no.  If you neglect to do this, you will likely receive an error when fitting the model that states that there are multi-collinearity issues with your data. This means that certain columns are highly correlated. For example, if you know that dependents_yes = 1 then you in turn know that dependents_no = 0.
# MAGIC 
# MAGIC * You can take one of two approaches when dropping variables. When using Pandas' get_dummies() function, you can simply set drop_first = True. An alternative approach, which is the one taken here, is to hand-select which variables to drop. In this notebook, we dropped the value that had a Kaplan-Meier survival probability curve most similar to the population.  While this is by no means necessary, it affords an intuitive interpretation as the baseline closely resembles the population.

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

# MAGIC %md
# MAGIC ## Fit the Cox Proportional Hazards Model
# MAGIC 
# MAGIC Similar to Kaplan-Meier, the first step when using the Lifelines library for Cox Proportional Hazards is to fit the model. 
# MAGIC * When fitting the model, we specified `alpha = 0.05`. This means that we will use a 95% confidence interval for our statistical tests.
# MAGIC 
# MAGIC * We provided three parameters to the fitted model:
# MAGIC   * `suvival_pd`: the dataframe used to fit the model
# MAGIC   * `Tenure:` the duration that a customer has been with the company (if still a subscriber) or was with the company prior to churning.
# MAGIC   * `Churn:` a Boolean indicating whether the customer is still a subscriber or not.

# COMMAND ----------

cph = CoxPHFitter(alpha=0.05)
cph.fit(survival_pd, 'tenure', 'churn')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Assess the Results of the Fitted Model
# MAGIC 
# MAGIC There are three key questions to ask when initially assessing the model:
# MAGIC 
# MAGIC **Is each covariate statistically significant?**
# MAGIC   * In the column labeled `p` below, it can be seen that the p-value for each column is below < 0.005. Therefore, each of the columns is statistically significant and safe to include. 
# MAGIC   * Similar to other forms of regression, in the event that a variable is not statistically significant, you can drop that column from your analysis or recategorize the corresponding values.
# MAGIC 
# MAGIC **How confident are we in the coefficient estimates?**
# MAGIC   * Upper and lower bounds for each coefficient and exp(coefficient) are provided in the model summary below (e.g. `exp(coef) lower 95%`, `exp(coef) upper 95%`).
# MAGIC   * Shown below the model summary, these bounds can also be viewed visually. Although it's not seen here, note that when viewing the bounds in this way, a variable can be deemed as not signficant when its corresponding box-and-whisker plot crosses the value 1.
# MAGIC 
# MAGIC **What is the effect of each covariate on the hazard ratio?**
# MAGIC   * Using internetService_DSL as an example, it's shown below that `coef = -0.22` and `exp(coef) = 0.80`. Referring back to the Cox Proportional Hazards equation, this means that a customer's hazard ratio decreases by a factor of `0.80` (versus the baseline) when they subscribe to DSL for their internet service.
# MAGIC   
# MAGIC Note that this output also includes several metrics that can be used when comparing models:
# MAGIC * `Concordance`
# MAGIC * `Partial AIC`
# MAGIC * `Log-likelihood Ratio`

# COMMAND ----------

cph.print_summary()

# COMMAND ----------

cph.plot(hazard_ratios=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify if the Model Adheres to the Proportional Hazard Assumption
# MAGIC 
# MAGIC * After assessing the results of the fitted model, the next step is to verify whether the model adheres to the proportional hazard assumption. We will do this using three methods:
# MAGIC   * Method 1: Statistical Test
# MAGIC   * Method 2: Schoenfield Residuals
# MAGIC   * Method 3: Log-log Kaplan-Meier Plots
# MAGIC 
# MAGIC * The results of using the first method -- a statistical test -- are shown below. As seen in the printout, Lifelines provides quite a bit of detail, including test results and recommendations. In the case of this model, it is seen that we violate the proportional hazard assumption for three of the four variables. This is illustrated by the p-values being less than 0.05 as well as the text below it. Of note, as hinted in the closing thoughts section of the Kaplan-Meier notebook, a red flag for this scenario is when you see the survival curves for a given covariate crossover each other when using Kaplan-Meier.

# COMMAND ----------

cph.check_assumptions(survival_pd,p_value_threshold=0.05)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Schoenfeld Residuals
# MAGIC 
# MAGIC * In addition to running a statistical test, it is helpful to also leverage a graphical output to assess the situation. This can be done using Schoenfeld residuals.
# MAGIC 
# MAGIC * In the output below, there are two plots for each variable. The difference between these two plots is the order in which the residual values are displayed: Rank tranformed time and KM-transformed time. No material difference is seen between these two types of plots for our model.
# MAGIC 
# MAGIC * The way to interpret these plots is similar to the way you would interpret residual plots for linear regression. In other words, when viewing this type of plot, we do not want to see any sort of pattern in the residuals. When no pattern is present, the black line in the middle will be relatively flat, indicating that the residuals are not correlated with time.
# MAGIC 
# MAGIC   * `internetService_DSL:` clear and consistent trend across time.
# MAGIC   
# MAGIC   * `onlineBackup_Yes:` most pronounced trend of the three variables.
# MAGIC   
# MAGIC   * `techSupport_Yes:` there is a bit of a pattern seen across time but the most profound impact comes from the points near the end of the timeline.

# COMMAND ----------

cph.check_assumptions(survival_pd,p_value_threshold=0.05,show_plots=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log-log Kaplan-Meier Plots
# MAGIC 
# MAGIC * Based on the results of the statistical test and the Schoenfeld residual plots, it's clear that our model violates the proportional hazards assumption many times over.
# MAGIC 
# MAGIC * To get another view of what the issue at play is here, we can use log-log Kaplan-Meier plots. As the name implies, this technique plots Kaplan-Meier curves on a log-log scale.
# MAGIC 
# MAGIC * It's important to note that this transformation of the Kaplan-Meier curves is simply altering the scale in which we view the data. This effectively smushes the data closer together, making it easier to assess.
# MAGIC 
# MAGIC * When the proportional hazard assumpton is not violated, the Kaplan-Meier curves in the log-log plot will appear parallel. This is in line with the plots reviewed above in the section "proportional hazard assumption."
# MAGIC 
# MAGIC * With the exception of internetService, it can be seen in the plots below that the Kaplan-Meier curves are mostly* parallel when log(timeline) is between 1 and 3 but less so when the log(timeline) is less than 1 or greater than 3.

# COMMAND ----------

# Fit the Kaplan-Meier model
kmf = KaplanMeierFitter()

T=telco_pd['tenure'] #duration
C=telco_pd['churn'].astype(float) #event observed

kmf.fit(T,C)

# COMMAND ----------

# Utility function for plotting
import matplotlib.pyplot as plt
def plot_km_loglog(col):
  ax = plt.subplot(111)
  for r in telco_pd[col].unique():
    ix = telco_pd[col] == r
    kmf.fit(T[ix], C[ix],label=r)
    kmf.plot_loglogs(ax=ax)

# COMMAND ----------

plot_km_loglog('onlineBackup')

# COMMAND ----------

plot_km_loglog('dependents')

# COMMAND ----------

plot_km_loglog('internetService')

# COMMAND ----------

plot_km_loglog('techSupport')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Closing Thoughts
# MAGIC 
# MAGIC * Cox Proportional Hazards is one of the most popular methods for Survival Analysis. This is in part due to CPH being a semi-parametric model, which means that your outcome variable does not need to adhere to a specific distribution. Further, CPH is known to fit various distributions fairly well.
# MAGIC 
# MAGIC * As discussed above, the key assumption that underpins this method is referred to as the proportional hazards assumption.  In the event that this assumption is violated, as is the case here, you have a number of options.
# MAGIC   1. `Leave the model as-is:` it's important to note that this assumption is most important when your objective is inference. If your objective is prediction, you can generally focus on the loss metric as your means for selecting a final model.
# MAGIC   
# MAGIC   2. `Stratify the existing model`: if there are a small number of variables causing the issue, you can stratify your model on those variables. The net result of using this approach is that you will have a different baseline hazard for each variable that you stratify the model on. Lifelines makes this easy to incorporate and further details can be found [here](https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#stratification).
# MAGIC   
# MAGIC   3. `Incoporate time-dependent variables`: as noted when reviewing the Cox Proportional Hazards equation, the partial hazard is time-independent. More specifically, the baseline hazard is a function of time and then the partial hazard is a function of the parameters. When a variable is truly time-dependent, it can be modeled as such using an interaction variable. This is referred to as the Extended Cox Proportional Hazards model. Further details can be found [here](https://lifelines.readthedocs.io/en/latest/Time%20varying%20survival%20regression.html#).
# MAGIC   
# MAGIC   4. `Use Cubic Splines or Piecewise Constant Hazards:` similar to other regression techniques, you can use cubic splines to add more flexibility to your model. You can think of this as splitting your model into multiple parts. For example, in our model above, it's clear that we see different patterns at certain points in time. What if there were 'knots' at these turning points, enabling your to alter your specification? In the context of Cox Proportional Hazards, using Cubic Splines results in the baseline hazard becoming parametric. Further details on implementing this in Lifelines can be found [here](https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#modeling-baseline-hazard-and-survival-with-parametric-models). For more information about Cubic splines in the more general case, the book [Introduction to Statistical Learning](https://statlearning.com/), and the corresponding Youtube videos, are a wonderful resource.
# MAGIC   
# MAGIC   5. `Use a Different Method:` although the Cox Proportional Hazards method can be extended in many ways, inclusive of a fully parametric form (e.g. Cubic Splines), it's important to know that other Survival Analysis methods exist. One alternative method is the Accelerated Failure Time model. The Accelerated Failure Time model is fully parametric and can be modeled using various distributions such as Exponential and Weibull. Note that this distribution refers to the shape of the outcome variable.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC * Thus far, we have built Survival Analysis models using non-parametric (Kaplan-Meier) and semi-parametric (Cox Proportional Hazards) methods.  We have also discussed the way in which a fully parametric Cox Proportional Hazard model can be used (e.g. Cubic Splines). For completeness, we will proceed with building a fully parametric model in our next notebook, using a method other than Cox Proportional Hazards. The method we will use is the Log-Logistic Accelerated Failure Time model.

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
