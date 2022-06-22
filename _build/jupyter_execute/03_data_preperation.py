#!/usr/bin/env python
# coding: utf-8

# # Data preperation
# 

# ## Conclusion form Data Analysis (EDA):
# 
# 
# * ratingprice and sellingprice have a very high correlation, therefore I would remove the column "ratingprice" from the dataset.
# 
# * code is not necessary, therefore I would remove the column "code" from the dataset.
# 
# * saledate is also unnecessary, therefore I would remove the column "saledate" from the dataset.
# 
# * there are almost only automatic cars in "drivetrain" - therefore I drop this feature.
# 
# * I have also removed the column "seller" to simplify the data-understanding of the project.
# 

# ### Load packages

# In[10]:


import pandas as pd
import numpy as np


# In[11]:


# import dataset and save it as df

df = pd.read_csv("car_prices.csv", on_bad_lines="skip")


# In[12]:


# drop missing vales (dataset is still big enough)

df = df.dropna()


# In[13]:


# rename colums for better understanding (as described above)

df = df.rename(columns={
"make" : "brand",
"body" : "type",
"trim" : "version",
"transmission" : "drivetrain",
"vin" : "code",
"odometer" : "miles",
"mmr" : "ratingprice"} 
    )


# In[14]:


# transform into lowercase

df["brand"] = df["brand"].str.lower()
df["model"] = df["model"].str.lower()
df["type"] = df["type"].str.lower()
df["drivetrain"] = df["drivetrain"].str.lower()
df["state"] = df["state"].str.lower()
df["version"] = df["version"].str.lower()
df["color"] = df["color"].str.lower()
df["interior"] = df["interior"].str.lower()
df["seller"] = df["seller"].str.lower()


# In[15]:


# transform into categorial variables

for cat in ["year", "brand", "model", "version", "type", "drivetrain", "code", "state", "condition", "color", "interior", "seller", "saledate"]:
    df[cat] = df[cat].astype("category")


# In[16]:


# drop irrelevant features

df = df.drop(["code", "ratingprice", "saledate", "drivetrain", "seller"], axis=1)


# In[17]:


df.info()


# In[18]:


df.head()


# In[ ]:


# export prepared dataset
from pathlib import Path  

filepath = Path('/Users/hendrikpfeifer/MLOps_SoSe22/car_prices_project/car_prices_clean.csv')  

filepath.parent.mkdir(parents=False, exist_ok=True)  

df.to_csv(filepath)  

