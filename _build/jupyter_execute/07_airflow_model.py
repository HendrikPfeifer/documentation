#!/usr/bin/env python
# coding: utf-8

# # Airflow - Model Building

# In[1]:


"""
    Regression model

Step 1) Import data
Step 2) Prepara data
Step 3) Fit model
Step 4) Store model

"""
#------------------------------------------------------
# Setup
import pandas as pd
import numpy as np
from joblib import dump

# Import Modules
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn import set_config
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#------------------------------------------------------
# Step 1) Import data 
df = pd.read_csv("/Users/hendrikpfeifer/airflow/dags/carprices_dag/df_clean.csv")

#------------------------------------------------------
# Step 2) Prepare data

# Convert to category

for cat in ["year", "brand", "model", "type", "state", "condition", "color", "interior"]:
    df[cat] = df[cat].astype("category")


# Transform numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
    ])

# Transform categorical features  
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

# Create Preprocessing Pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, selector(dtype_exclude="category")),
    ('cat', categorical_transformer, selector(dtype_include="category"))
        ])


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

#X.info()
#print("Missing values:",X.isnull().any(axis = 1).sum())

# Define Label
y = df["sellingprice"]


# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create Pipeline with model
lm_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lm', LinearRegression())
                        ])



#------------------------------------------------------
# Step 3) Fit model

# Train the model
lm_model.fit(X_train, y_train)


# Test model
y_pred = lm_model.predict(X_test)

# R2-score
#r2_score(y_test, y_pred)

#MSE
#mean_squared_error(y_test, y_pred)
#mean_squared_error(y_test, y_pred, squared=False)

#MAE
#mean_absolute_error(y_test, y_pred)

#------------------------------------------------------
# Step 4) Store model

dump(lm_model, "/Users/hendrikpfeifer/airflow/dags/carprices_dag/my_linear_model.joblib")




