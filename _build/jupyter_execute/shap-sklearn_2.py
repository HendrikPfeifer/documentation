#!/usr/bin/env python
# coding: utf-8

# # SHAP SKlearn 2

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

#from sklearn_pandas import DataFrameMapper


# In[2]:


# Import Dataset

df = pd.read_csv("car_prices_clean.csv", on_bad_lines="skip")
df = df.drop(columns=['Unnamed: 0'])


# In[3]:


df.head()


# In[4]:


df.info()


# In[13]:


mapper = DataFrameMapper(num + cat, df_out=True)
preprocessed_X_train = mapper.fit_transform(X_train)
preprocessed_X_train = sm.add_constant(preprocessed_X_train)
reg = fit(y_train, preprocessed_X_trai)



# In[6]:


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


# In[8]:


from sklearn.model_selection import train_test_split

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[10]:


catagorical_features = ["year", "brand", "model", "type", "state", "condition", "color", "interior", "seller"]
numerical_features = [c for c in X_train.columns if c not in catagorical_features]
cat = [([c], [OneHotEncoder()]) for c in catagorical_features]
num = [([n], [SimpleImputer(), StandardScaler()]) for n in numerical_features]


# In[7]:


X.info()

print("Missing values:",X.isnull().any(axis = 1).sum())


# In[9]:


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


# In[12]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)


# In[32]:


# Explain the linear model

explainer = shap.Explainer(model.predict, X_train)
#shap_values = explainer(X_test)
#X_test_array = X_test.toarray() # we need to pass a dense version for the plotting functions

