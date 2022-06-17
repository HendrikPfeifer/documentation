#!/usr/bin/env python
# coding: utf-8

# ## Setup

# In[32]:


import keras_tuner as kt

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorboard import notebook
from tensorboard.plugins.hparams import api as hp

import datetime

get_ipython().run_line_magic('load_ext', 'tensorboard')


# ## Data import

# In[16]:


df = pd.read_csv("car_prices_clean.csv", on_bad_lines="skip")
df = df.drop(columns=['Unnamed: 0', "seller"])


# In[17]:


dummies = pd.get_dummies(df[["brand", "model", "type", "state", "color", "interior"]])


# In[18]:


# make target variable
#y = df.pop('sellingprice')

y = df['sellingprice']


# In[19]:


X_numerical = df.drop(["sellingprice", "brand", "model", "type", "state", "color", "interior"], axis=1).astype('float64')


# In[20]:


list_numerical = X_numerical.columns


# In[21]:


# Create all features
X = pd.concat([X_numerical, dummies], axis=1)


# In[22]:


train_ratio = 0.80
test_ratio = 0.10
val_ratio = 0.10

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=10)


# In[23]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio/(train_ratio+test_ratio), random_state=10)


# In[24]:


print(X_train.shape)
print(X_test.shape)
print(X_val.shape)


# In[25]:


scaler = StandardScaler().fit(X_train[list_numerical]) 

X_train[list_numerical] = scaler.transform(X_train[list_numerical])
X_test[list_numerical] = scaler.transform(X_test[list_numerical])
X_val[list_numerical] = scaler.transform(X_val[list_numerical])


# ## Define search space

# In[26]:


def build_model(hp):

    model = keras.Sequential()
    
    model.add(
        layers.Dense(
            # Define the hyperparameter.
            units = hp.Int("units", min_value=32, 
                                    max_value=512, 
                                    step=32),
            activation = "relu",
        )
    )
    model.add(layers.Dense(1))
    
    model.compile(
        optimizer="adam", loss="mse", metrics=["mean_absolute_error"],
    
    )
    return model


# In[27]:


build_model(kt.HyperParameters())


# In[36]:


def build_model(hp):
    
    model = keras.Sequential()
    
    model.add(
        layers.Dense(
            # Tune number of units.
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            # Tune the activation function to use.
            activation=hp.Choice("activation", ["relu", "tanh"]),
        )
    )
    
    # Tune whether to use dropout.
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(1))
    
    # Define the optimizer learning rate as a hyperparameter.
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mean_absolute_error"],
    )
    
    return model


# In[37]:


build_model(kt.HyperParameters())


# In[38]:


tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective="val_mean_absolute_error",
    max_trials=5,
    executions_per_trial=2,
    overwrite=True,
    directory="tmp",
    project_name="car_hyperparameter",
)


# In[39]:


tuner.search_space_summary()


# In[40]:


# Create TensorBoard folders
log_dir = "tmp/tb_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tuner.search(
    X_train,
    y_train,
    epochs=2,
    validation_data=(X_val, y_val),
    # Use the TensorBoard callback.
    # The logs will be write to "/tmp/tb_logs".
    callbacks=[keras.callbacks.TensorBoard(log_dir=log_dir)],
)


# In[49]:


get_ipython().run_line_magic('tensorboard', '--logdir /Users/hendrikpfeifer/MLOps_SoSe22/car_prices_project/tmp')

