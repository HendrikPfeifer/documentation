#!/usr/bin/env python
# coding: utf-8

# # SHAP SKlearn 1

# In[1]:


import sklearn 
import pandas as pd
import numpy as np

import shap
shap.initjs()

# Modules
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn import set_config
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


# Import Dataset

df = pd.read_csv("car_prices_clean.csv", on_bad_lines="skip")
df = df.drop(columns=['Unnamed: 0'])


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


for cat in ["year", "brand", "model", "type", "state", "condition", "color", "interior", "seller"]:
    df[cat] = df[cat].astype("category")


# In[6]:


# for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
    ])


# In[7]:


# for categorical features  
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])


# In[8]:


# Pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, selector(dtype_exclude="category")),
    ('cat', categorical_transformer, selector(dtype_include="category"))
        ])


# In[9]:


# Select features for multiple regression
features= [
 'miles',
 'brand',
 'model',
 'type',
 'condition',
 'color'
  ]
X = df[features]

# Create response
y = df["sellingprice"]


# In[10]:


X.info()

print("Missing values:",X.isnull().any(axis = 1).sum())


# In[12]:


from sklearn.model_selection import train_test_split

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[35]:


X_train.shape[0]


# In[49]:


print("Model coefficients:\n")
for i in range(X_train.shape[1]):
    print(X_train.columns[0], "=", X_train.iloc[0])


# In[14]:


from sklearn.linear_model import LinearRegression

# Create pipeline with model

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lm', LinearRegression())
                        ])


# In[15]:


model.fit(X_train, y_train)


# In[32]:


# Explain the linear model

explainer = shap.Explainer(model.predict, X_train)
#shap_values = explainer(X_test)
#X_test_array = X_test.toarray() # we need to pass a dense version for the plotting functions

