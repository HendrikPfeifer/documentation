#!/usr/bin/env python
# coding: utf-8

# # Scikit-Learn
# 
# #### Linear Regression-, KNN-Model and Random Forest Regressor for Multiple Regression 
# 

# ## Load packages

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant  

sns.set_theme()


# ## Import Dataset

# In[2]:


raw_dataset = pd.read_csv("car_prices.csv", on_bad_lines="skip")


# In[3]:


df = raw_dataset.copy()


# ## Data inspection

# In[4]:


df.head(2)


# In[5]:


df.info()


# In[6]:


print(df.isnull().sum())


# ## Data transformation

# In[7]:


# drop column with too many missing values
df = df.drop(['transmission'], axis=1)

# drop remaining row with one missing value
df = df.dropna()


# In[8]:


# Drop irrelevant features
df = df.drop(['trim', 'vin', 'mmr', 'saledate'], axis=1)


# In[9]:


print(df.isnull().sum())


# In[10]:


# rename columns

df = df.rename(columns={
"make" : "brand",
"body" : "type",
"odometer" : "miles"} 
    )


# In[11]:


df.info()


# In[12]:


# transform into lowercase

df["brand"] = df["brand"].str.lower()
df["model"] = df["model"].str.lower()
df["type"] = df["type"].str.lower()


# # Categorial or numeric?
# 
# * year = categorial
# * brand = categorial
# * model = categorial
# * type = categorial
# * state = categorial
# * condition = categorial
# * miles = numeric
# * color = categorial
# * interior = categorial
# * seller = categorial
# * ratingprice = numeric
# * sellingprice = numeric
# * saledate = categorial

# In[13]:


# transform to categorical:

for cat in ["year", "brand", "model", "type", "state", "condition", "color", "interior", "seller"]:
    df[cat] = df[cat].astype("category")


# In[14]:


df.info()


# In[15]:


# summary statistics for all categorical columns
df.describe(include=['category']).transpose()


# ## Data preprocessing pipeline

# In[16]:


# Modules
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn import set_config
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# In[17]:


# for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
    ])


# In[18]:


# for categorical features  
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])


# In[19]:


# Pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, selector(dtype_exclude="category")),
    ('cat', categorical_transformer, selector(dtype_include="category"))
        ])


# In[20]:


df.head()


# ## Modeling

# # Multiple Regression

# In[21]:


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

X.info()
print("Missing values:",X.isnull().any(axis = 1).sum())

# Create response
y = df["sellingprice"]


# In[22]:


from sklearn.model_selection import train_test_split

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[23]:


from sklearn.linear_model import LinearRegression

# Create pipeline with model
lm_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lm', LinearRegression())
                        ])


# In[24]:


# show pipeline
set_config(display="diagram")
# Fit model
lm_pipe.fit(X_train, y_train)


# In[25]:


y_pred = lm_pipe.predict(X_test)


# In[26]:


from sklearn.metrics import r2_score

r2_score(y_test, y_pred)


# In[27]:


from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, y_pred)


# In[28]:


mean_squared_error(y_test, y_pred, squared=False)


# In[29]:


from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, y_pred)


# In[74]:


# Create a new GDP value
X_new = pd.DataFrame({
    "year": [2015],
    "brand": "kia",
    "model": "sorento",
    "type": "suv",
    "state": "ca",
    "condition": [5.0],
    "miles": [16639.0],
    "color": "white",
    "interior": "black",
    "seller": "kia motors america, inc",
})


# In[75]:


X_new


# In[76]:


my_prediction = lm_pipe.predict(X_new)


# In[77]:


df_prediction = pd.DataFrame({"pred": my_prediction})


# In[78]:


df_prediction


# In[72]:


sample = {
    "year": [2015],
    "brand": "kia",
    "model": "sorento",
    "type": "suv",
    "state": "ca",
    "condition": [5.0],
    "miles": [16639.0],
    "color": "white",
    "interior": "black",
    "seller": "kia motors america, inc",
}


# In[52]:


# KNN
from sklearn.neighbors import KNeighborsRegressor as KNR

# Create pipeline with model
knn_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('KNN Regression', KNR(n_neighbors=5))
                        ])


# In[53]:


# show pipeline
set_config(display="diagram")
# Fit model
knn_pipe.fit(X_train, y_train)


# In[54]:


y_pred = knn_pipe.predict(X_test)


# In[55]:


#save model
from joblib import dump

dump(knn_pipe, "knn_model.joblib")


# In[56]:


r2_score(y_test, y_pred)


# In[57]:


mean_squared_error(y_test, y_pred)


# In[58]:


mean_squared_error(y_test, y_pred, squared=False)


# In[59]:


mean_absolute_error(y_test, y_pred)


# In[79]:


my_prediction = knn_pipe.predict(X_new)


# In[80]:


df_prediction_knn = pd.DataFrame({"pred": my_prediction})


# In[81]:


df_prediction_knn


# In[60]:


# RandomForest
from sklearn.ensemble import RandomForestRegressor as RFR

# Create pipeline with model
rf_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('Random Forest', RFR(n_estimators=50, max_depth=5))
                        ])


# In[61]:


# show pipeline
set_config(display="diagram")
# Fit model
rf_pipe.fit(X_train, y_train)


# In[62]:


#save model
from joblib import dump

dump(rf_pipe, "rf_model.joblib")


# In[63]:


y_pred = rf_pipe.predict(X_test)


# In[64]:


r2_score(y_test, y_pred)


# In[65]:


mean_squared_error(y_test, y_pred)


# In[66]:


mean_squared_error(y_test, y_pred, squared=False)


# In[67]:


mean_absolute_error(y_test, y_pred)

