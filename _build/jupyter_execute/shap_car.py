#!/usr/bin/env python
# coding: utf-8

# # SHAP (test)

# In[1]:


from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Flatten, Concatenate, concatenate, Dropout, Lambda
from keras.models import Model
from keras.layers.embeddings import Embedding
from tqdm import tqdm
import shap

import tensorflow as tf
from tensorflow.keras import layers

import pandas as pd

import numpy as np

# print the JS visualization code to the notebook
shap.initjs()


# In[2]:


# import dataset 
df = pd.read_csv("car_prices_clean.csv")

df = df.drop(columns=['Unnamed: 0'])


# In[3]:


df.info()


# In[4]:


# Make a dictionary with int64 features as keys and np.int32 as values
int_32 = dict.fromkeys(df.select_dtypes(np.int8).columns, np.int32)
# Change all columns from dictionary
df = df.astype(int_32)

# Make a dictionary with float64 columns as keys and np.float32 as values
float_32 = dict.fromkeys(df.select_dtypes(np.float64).columns, np.float32)
df = df.astype(float_32)


# Convert to categorical

# make a list of all categorical variables
cat_convert = ["brand", "model", "type", "state", "color", "interior", "seller"]

# convert variables
for i in cat_convert:
    df[i] = df[i].astype("string")

# Convert to category
df['year'] = df['year'].astype("category")
df['condition'] = df['condition'].astype("category")




# In[5]:


df.info()


# In[ ]:





# In[6]:


X = df.drop(columns=["sellingprice"])


# In[7]:


X.head()


# In[8]:


model = tf.keras.models.load_model('my_car_model-mean-absolute')


# In[9]:


sample = {
    "year": 2015,
    "brand": "kia",
    "model": "sorento",
    "type": "suv",
    "state": "ca",
    "condition": 5.0,
    "miles": 16639.0,
    "color": "white",
    "interior": "black",
    "seller": "kia motors america, inc",
}


# In[10]:


input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}


# In[11]:


predictions = model.predict(input_dict)


# In[12]:


predictions


# In[ ]:


X = df


# In[25]:


df.to_numpy()


# In[76]:


def f(X):
    return model.predict([X[:,i] for i in range(X.shape[1])]).flatten()




# In[81]:


X.shape[1].flatten()


# In[ ]:





# In[ ]:


for i in range:
    "year": 2015,
    "brand": "kia",
    "model": "sorento",
    "type": "suv",
    "state": "ca",
    "condition": 5.0,
    "miles": 16639.0,
    "color": "white",
    "interior": "black",
    "seller": "kia motors america, inc",
    


# In[77]:


X.info()


# In[88]:


explainer = shap.KernelExplainer(model.predict(input_dict), X.iloc[:1,:])


# In[ ]:


shap_values = explainer.shap_values(X.iloc[5,:], nsamples=60) #von einer Person ausgesucht (299) und 500 samples ->erzeugung von Shap-Values
shap.force_plot(explainer.expected_value, shap_values, X.iloc[5,:])

