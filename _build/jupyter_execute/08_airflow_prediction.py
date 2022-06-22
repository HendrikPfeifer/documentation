#!/usr/bin/env python
# coding: utf-8

# # Airflow - Prediction

# In[1]:


"""
    Make prediction

Step 1) Import model 
Step 2) Create new data
Step 3) Make prediction
Step 4) Store prediction

"""
#------------------------------------------------------
# Setup
import pandas as pd
from joblib import load


#------------------------------------------------------
# Step 1) Import model 
lm_model = load("/Users/hendrikpfeifer/airflow/dags/carprices_dag/my_linear_model.joblib")

#------------------------------------------------------
# Step 2) Make new data

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
})

#------------------------------------------------------
# Step 3) Make prediction

# Make prediction
my_prediction = lm_model.predict(X_new)

#------------------------------------------------------
# Step 4) Save prediction

# Save prediction as dataframe 
df_prediction = pd.DataFrame({"pred": my_prediction})

# Store predictions as csv
df_prediction.to_csv("/Users/hendrikpfeifer/airflow/dags/carprices_dag/my_prediction.csv")



