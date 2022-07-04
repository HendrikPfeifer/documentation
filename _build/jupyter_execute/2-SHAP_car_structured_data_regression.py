#!/usr/bin/env python
# coding: utf-8

# # SHAP 
# ## with structured data regression
# 
# **Explainable AI with TensorFlow, Keras and SHAP**
# 
# SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions. [Learn more](https://shap.readthedocs.io/en/latest/index.html)  
# 
# *This is mainly based on the Keras tutorial ["Structured data classification from scratch"](https://keras.io/examples/structured_data/structured_data_classification_from_scratch/) by François Chollet and ["Census income classification with Keras"](https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/neural_networks/Census%20income%20classification%20with%20Keras.html#Census-income-classification-with-Keras) by Scott Lundberg.*

# ## Setup

# In[1]:


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# In[2]:


import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import shap

tf.__version__


# In[3]:


# print the JS visualization code to the notebook
shap.initjs()


# ## Data

# ### Data import

# 
# - Let's download the data and load it into a Pandas dataframe:

# In[4]:


df = pd.read_csv("car_prices_clean.csv", on_bad_lines="skip")
df = df.drop(columns=['Unnamed: 0', "seller"])


# In[5]:


df.head()


# In[6]:


df.info()


# ### Data preparation

# ### Create labels and features
# 
# We need to encode our categorical features as one-hot numeric features (dummy variables):

# In[7]:


dummies = pd.get_dummies(df[["year","brand", "model", "type", "state", "condition", "color", "interior"]])


# In[8]:


dummies.info()


# In[9]:


print(dummies.head())


# In[10]:


# make target variable

y = df['sellingprice']


# In[11]:


X_numerical = df.drop(["sellingprice","year", "brand", "model", "type", "state", "condition", "color", "interior"], axis=1).astype('float64')


# In[12]:


list_numerical = X_numerical.columns
list_numerical


# In[13]:


# Create all features

X = pd.concat([X_numerical, dummies], axis=1)
X.info()


# ### Data splitting

# - Let's split the data into a training and test set

# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)


# In[15]:


X_train.head()


# ## Model

# Now we can build the model using the Keras sequential API:

# In[17]:


model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
  ])


# In[18]:


model.compile(optimizer="adam", 
              loss ="mse", 
              metrics=["mean_absolute_error"])


# In[19]:


# will stop training when there is no improvement in 3 consecutive epochs

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)


# In[20]:


model.fit(X_train, y_train, 
         epochs=10,
         validation_data=(X_test, y_test), 
         callbacks=[callback]
         )


# - Configure the model with Keras Model.compile:

# Let's visualize our connectivity graph:

# In[21]:


tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")


# In[22]:


loss, accuracy = model.evaluate(X_test, y_test)

print("MAE:", accuracy)


# ## Perform inference

# - The model you have developed can now classify a row from a CSV file directly after you've included the preprocessing layers inside the model itself. Let's demonstrate the process:

# - Save the heart diseases classification model

# In[23]:


model.save('shap_car_model-2')


# - Load model

# In[24]:


reloaded_model = tf.keras.models.load_model('shap_car_model-2')


# In[25]:


predictions = reloaded_model.predict(X_train)


# In[26]:


predictions


# ## SHAP

# We use our model and a selection of 50 samples from the dataset to represent “typical” feature values (the so called background distribution).

# In[27]:


explainer = shap.KernelExplainer(model, X_train.iloc[:50,:])


# Now we use 500 perterbation samples to estimate the SHAP values for a given prediction (at index location 20). Note that this requires 500 * 50 evaluations of the model.

# In[28]:


shap_values = explainer.shap_values(X_train.iloc[20,:], nsamples=500)


# The so called force plot below shows how each feature contributes to push the model output from the base value (the average model output over the training dataset we passed) to the model output. Features pushing the prediction higher are shown in red, those pushing the prediction lower are in blue. To learn more about force plots, take a look at this [Nature BME paper](https://www.nature.com/articles/s41551-018-0304-0.epdf?author_access_token=vSPt7ryUfdSCv4qcyeEuCdRgN0jAjWel9jnR3ZoTv0PdqacSN9qNY_fC0jWkIQUd0L2zaj3bbIQEdrTqCczGWv2brU5rTJPxyss1N4yTIHpnSv5_nBVJoUbvejyvvjrGTb2odwWKT2Bfvl0ExQKhZw%3D%3D) from Lundberg et al. (2018).

# In[32]:


shap.force_plot(explainer.expected_value[0], shap_values[0], X_train.iloc[20,:])


# ### Explain many predictions

# If we take many force plot explanations such as the one shown above, rotate them 90 degrees, and then stack them horizontally, we can see explanations for an entire dataset (see content below). Here, we repeat the above explanation process for 50 individuals.
# 
# To understand how a single feature effects the output of the model we can plot the SHAP value of that feature vs. the value of the feature for all the examples in a dataset. Since SHAP values represent a feature's responsibility for a change in the model output, the plot below represents the change in the dependent variable. Vertical dispersion at a single value of represents interaction effects with other features. 

# In[30]:


shap_values50 = explainer.shap_values(X_train.iloc[50:100,:], nsamples=500)


# In[31]:


shap.force_plot(explainer.expected_value[0], shap_values50[0], X_train.iloc[50:100,:])

