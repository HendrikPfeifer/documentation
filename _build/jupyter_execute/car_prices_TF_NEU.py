#!/usr/bin/env python
# coding: utf-8

# # TensorFlow
# 
# #### Models for: Linear Regression (Simple Regression), Linear Regression (Multiple Regression) and DNN Regression (Multiple Regression)

# ## Setup

# In[2]:


# Load packages

get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)

sns.set_theme(style="ticks", color_codes=True)


# ## Data preparation

# In[3]:


from car_prices_data_prep import *


# In[4]:


df.info()


# ## Simple regression

# In[9]:


# Select features for simple regression
features = ['miles']
X = df[features]

X.info()
print("Missing values:",X.isnull().any(axis = 1).sum())

# Create response
y = df["sellingprice"]


# ## Data splitting

# In[10]:


from sklearn.model_selection import train_test_split

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Linear regression

# In[11]:


lm = tf.keras.Sequential([
    layers.Dense(units=1, input_shape=(1,))
])

lm.summary()


# In[12]:


# untrained model for first 10 values
lm.predict(X_train[:10])


# In[13]:


lm.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


# In[15]:


get_ipython().run_cell_magic('time', '', 'history = lm.fit(\n    X_train, y_train,\n    epochs=10,\n    # suppress logging\n    verbose=0,\n    # Calculate validation results on 20% of the training data\n    validation_split = 0.1)\n')


# In[16]:


y_train


# In[17]:


# Calculate R squared
from sklearn.metrics import r2_score

y_pred = lm.predict(X_train).astype(np.int64)
y_true = y_train.astype(np.int64)

r2_score(y_train, y_pred)  


# In[18]:


# slope coefficient
lm.layers[0].kernel


# In[19]:


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


# In[20]:


def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.xlabel('Epoch')
  plt.ylabel('Error [price]')
  plt.legend()
  plt.grid(True)


# In[21]:


plot_loss(history)


# In[22]:


test_results = {}

test_results['lm'] = lm.evaluate(
    X_test,
    y_test, verbose=0)

test_results


# In[23]:


x = tf.linspace(0.0, 6200, 6201)
y = lm.predict(x)

y


# In[27]:


def plot_area(x, y):
  plt.scatter(X_train['miles'], y_train, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('miles')
  plt.ylabel('sellingprice')
  plt.legend()


# In[28]:


plot_area(x,y)


# ## Multiple Regression

# In[5]:


# Select all relevant features
features= [
 'miles',
 'brand',
 'model',
 'type',
 'condition',
 'color'
  ]
X = df[features]

# Convert categorical to numeric
X = pd.get_dummies(X, columns=["brand", "model", "type", "condition", "color"])


X.info()
print("Missing values:",X.isnull().any(axis = 1).sum())

# Create response
y = df["sellingprice"]


# In[6]:


from sklearn.model_selection import train_test_split

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


lm_2 = tf.keras.Sequential([
    layers.Dense(units=1, input_shape=(928,))
])

lm_2.summary()


# In[49]:


lm_2.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


# In[9]:


get_ipython().run_cell_magic('time', '', 'history = lm_2.fit(\n    X_train, y_train,\n    epochs=10,\n    # suppress logging\n    verbose=0,\n    # Calculate validation results on 20% of the training data\n    validation_split = 0.1)\n')


# In[52]:


# Calculate R squared
from sklearn.metrics import r2_score

y_pred = lm_2.predict(X_train).astype(np.int64)
y_true = y_train.astype(np.int64)

r2_score(y_train, y_pred)  


# In[53]:


# slope coefficients
lm_2.layers[0].kernel


# In[54]:


plot_loss(history)


# In[55]:


test_results['lm_2'] = lm_2.evaluate(
    X_test, y_test, verbose=0)


# ## DNN regression

# In[15]:


dnn_model = keras.Sequential([
      layers.Dense(units=1, input_shape=(928,)),
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

dnn_model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))


# In[16]:


get_ipython().run_cell_magic('time', '', 'history = dnn_model.fit(\n    X_train, y_train,\n    epochs=10,\n    # suppress logging\n    verbose=0,\n    # Calculate validation results on 20% of the training data\n    validation_split = 0.1)\n')


# In[17]:


X_train


# In[59]:


# Calculate R squared
from sklearn.metrics import r2_score

y_pred = dnn_model.predict(X_train).astype(np.int64)
y_true = y_train.astype(np.int64)

r2_score(y_train, y_pred)  


# In[60]:


plot_loss(history)


# In[61]:


test_results['dnn_model'] = dnn_model.evaluate(
    X_test, y_test, verbose=0)


# ## Performance comparision

# In[62]:


pd.DataFrame(test_results, index=['Mean absolute error [sellingprice]']).T

