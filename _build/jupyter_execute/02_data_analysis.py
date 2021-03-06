#!/usr/bin/env python
# coding: utf-8

# # Data analysis
# 
# ### Exploratory data analysis (EDA) to get an better overview over the "used car prices"-dataset.

# ### Load packages

# In[2]:


import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt


# ### Import dataset

# In[3]:


# import dataset and save it as df

df = pd.read_csv("car_prices.csv", on_bad_lines="skip")

# on_bad_lines="skip" otherwise it caused a problem


# In[4]:


# show first five rows to check if the dataset is imported correctly 

df.head(5)


# - year = year the car was build
# - make = brand of the car           
# - model = cars model            
# - trim = cars version            
# - body = cars type           
# - transmission = cars drivetrain     
# - vin = code            
# - state = state where the car was sold           
# - condition = condition of the car 0.0 - 5.0       
# - odometer = miles of the car      
# - color = cars color           
# - interior = interior color         
# - seller = seller           
# - mmr = ratingprice              
# - sellingprice = sellingprice     
# - saledate = date of sale 

# In[23]:


# print how many observations and columns the dataset exists of

print(f"We have {len(df.index):,} observations and {len(df.columns)} columns in our dataset.")


# In[24]:


# overview

df.info()


# In[25]:


# print the names of all 16 coulmns

df.columns


# In[26]:


# print datatype of the variables

df.dtypes


# In[27]:


# print missing values

df.isna().sum()

# in transmission are relatively many missing values


# In[28]:


# show missing values (missing values - if present - will be displayed in yellow )
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis');


# In[29]:


# drop missing vales (dataset is still big enough)

df = df.dropna()


# In[30]:


# show missing values (missing values - if present - will be displayed in yellow )
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis');


# In[31]:


# show if there are still missing values

df.isna().sum()


# In[32]:


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


# In[33]:


df.info()


# In[34]:


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


# In[35]:


df.head(2)


# # Categorial or numeric?
# 
# * year = categorial
# * brand = categorial
# * model = categorial
# * version = categorial
# * type = categorial
# * drivetrain = categorial
# * code = categorial
# * state = categorial
# * condition = categorial
# * miles = numeric
# * color = categorial
# * interior = categorial
# * seller = categorial
# * ratingprice = numeric
# * sellingprice = numeric
# * saledate = categorial

# In[36]:


# transform into categorial variables

for cat in ["year", "brand", "model", "version", "type", "drivetrain", "code", "state", "condition", "color", "interior", "seller", "saledate"]:
    df[cat] = df[cat].astype("category")


# In[37]:


df.dtypes


# In[38]:


df.describe(include="category").T


# In[39]:


# crating variable list for numeric and categorial variables

# list of all numerical data
list_num = df.select_dtypes(include=[np.number]).columns.tolist()

# list of all categorical data
list_cat = df.select_dtypes(include=['category']).columns.tolist()

print(list_num, list_cat)


# ## Categorical Data

# In[40]:


# print plots for top 10 of each variable

for i in list_cat:

    TOP_10 = df[i].value_counts().iloc[:10].index

    g = sns.catplot(y=i, 
            kind="count", 
            palette="ch:.25", 
            data=df,
            order = TOP_10)    
    
    plt.title(i)
    plt.show();


# In[41]:


# Numercial gruped by categorical
# median
for i in list_cat:
    print(df.groupby(i).median().round(2).T)


# ## Numerical data

# In[42]:


# summary of numerical attributes
df.describe().round(2).T


# In[43]:


# histograms
df.hist(figsize=(20, 15));


# In[44]:


sns.set_theme(style="ticks", color_codes=True)


# In[45]:


sns.pairplot(df);


# In[46]:


sns.scatterplot(data=df, x="miles", y="sellingprice")


# In[47]:


sns.histplot(data=df, x="ratingprice")


# In[48]:


sns.histplot(data=df, x="sellingprice")


# # Relationships
# ## Correlation
# 
# Detect the relationships between variables
# 

# In[49]:


# inspect correlation

print(df.corr())
sns.heatmap(df.corr())


# # Conclusion:
# 
# 
# * ratingprice and sellingprice have a very high correlation, therefore I would remove the column "ratingprice" from the dataset.
# * code is not necessary, therefore I would remove the column "code" from the dataset.
# * saledate is also unnecessary, therefore I would remove the column "saledate" from the dataset.
# 
# * there are almost only automatic cars in "drivetrain" - not sure if I need this column for my model
