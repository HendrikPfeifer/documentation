#!/usr/bin/env python
# coding: utf-8

# # Airflow - Data Preperation

# In[1]:


""" 
    Data preparation

Step 1) Import data with pandas
Step 2) Make some data corrections
Step 3) Save data as csv to local folder

"""
#------------------------------------------------------
# Setup
import pandas as pd


#------------------------------------------------------
# Step 1: Import data from local folder
df = pd.read_csv("/Users/hendrikpfeifer/airflow/dags/carprices_dag/car_prices.csv", on_bad_lines="skip")

#------------------------------------------------------
# Step 2: Data cleaning 

# drop column with too many missing values
df = df.drop(['transmission'], axis=1)

# drop remaining row with one missing value
df = df.dropna()

# Drop irrelevant features
df = df.drop(['trim', 'vin', 'mmr', 'saledate', 'seller'], axis=1)

# rename columns
df = df.rename(columns={
"make" : "brand",
"body" : "type",
"odometer" : "miles"} 
    )

# transform into lowercase
df["brand"] = df["brand"].str.lower()
df["model"] = df["model"].str.lower()
df["type"] = df["type"].str.lower()



#------------------------------------------------------

# Step 3: Save data to current working directory
df.to_csv('/Users/hendrikpfeifer/airflow/dags/carprices_dag/df_clean.csv', index=False)




