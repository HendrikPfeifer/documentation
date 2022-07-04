#!/usr/bin/env python
# coding: utf-8

# # Keras 
# 
# ##### Keras Model

# ## Load Packages

# In[1]:


import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers

import keras_tuner as kt


tf.__version__


# In[2]:


# import dataset 
df = pd.read_csv("car_prices.csv", on_bad_lines="skip")


# In[3]:


df.head()


# In[38]:


df.info()


# In[39]:


# drop column with too many missing values
df = df.drop(['transmission'], axis=1)


# In[40]:


# drop remaining row with one missing value
df = df.dropna()


# In[41]:


# Drop irrelevant features
df = df.drop(['trim', 'vin', 'mmr', 'saledate', 'seller'], axis=1)


# In[42]:


# rename columns
df = df.rename(columns={
"make" : "brand",
"body" : "type",
"odometer" : "miles"} 
    )


# In[43]:


# transform into lowercase
df["brand"] = df["brand"].str.lower()
df["model"] = df["model"].str.lower()
df["type"] = df["type"].str.lower()


# ## Define label

# In[44]:


y_label = 'sellingprice'


# ## Data format

# In[45]:


# Make a dictionary with int64 features as keys and np.int32 as values
int_32 = dict.fromkeys(df.select_dtypes(np.int64).columns, np.int32)
# Change all columns from dictionary
df = df.astype(int_32)

# Make a dictionary with float64 columns as keys and np.float32 as values
float_32 = dict.fromkeys(df.select_dtypes(np.float64).columns, np.float32)
df = df.astype(float_32)


# In[46]:


int_32


# In[47]:


# Convert to categorical

# make a list of all categorical variables
cat_convert = ["brand", "model", "type", "state", "color", "interior"]

# convert variables
for i in cat_convert:
    df[i] = df[i].astype("string")


# In[48]:


# Convert to category

df['year'] = df['year'].astype("category")
df['condition'] = df['condition'].astype("category")


# In[49]:


df.info()


# In[50]:


# Make list of all numerical data (except label)
list_num = df.drop(columns=[y_label]).select_dtypes(include=[np.number]).columns.tolist()

# Make list of all categorical data which is stored as integers (except label)
list_cat_int = df.drop(columns=[y_label]).select_dtypes(include=['category']).columns.tolist()

# Make list of all categorical data which is stored as string (except label)
list_cat_string = df.drop(columns=[y_label]).select_dtypes(include=['string']).columns.tolist()


# In[51]:


list_num


# In[52]:


list_cat_int


# In[53]:


df.info()


# ## Data Splitting

# In[55]:


# Make test data
df_test = df.sample(frac=0.2, random_state=1337)

# Create training data
df_train = df.drop(df_test.index)


# In[56]:


print(
    "Using %d samples for training and %d for validation"
    % (len(df_train), len(df_test))
)


# ## Transform to Tensors

# In[57]:


# Define a function to create our tensors

def dataframe_to_dataset(dataframe, shuffle=True, batch_size=32):
    df = dataframe.copy()
    labels = df.pop(y_label) #pick y_label and delete
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels)) #ds for tensors
    if shuffle:
        ds = ds.shuffle(buffer_size=10000) #len(dataframe)
    ds = ds.batch(batch_size)
    df = ds.prefetch(batch_size)
    return ds


# In[58]:


batch_size = 32

ds_train = dataframe_to_dataset(df_train, shuffle=True, batch_size=batch_size)
ds_test = dataframe_to_dataset(df_test, shuffle=True, batch_size=batch_size)


# In[59]:


ds_train


# # Feature preprocessing
# ### Numerical preprocessing function

# In[60]:


# Define numerical preprocessing function
def get_normalization_layer(name, dataset):
    
    # Create a Normalization layer for our feature
    normalizer = layers.Normalization(axis=None)

    # Prepare a dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    return normalizer


# ### Categorical preprocessing function

# In[61]:


def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
  
  # Create a layer that turns strings into integer indices.
  if dtype == 'string':
    index = layers.StringLookup(max_tokens=max_tokens)
  # Otherwise, create a layer that turns integer values into integer indices.
  else:
    index = layers.IntegerLookup(max_tokens=max_tokens) #, output_mode='multi_hot'

  # Prepare a `tf.data.Dataset` that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the set of possible values and assign them a fixed integer index.
  index.adapt(feature_ds)

  # Encode the integer indices.
  encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())

  # Apply multi-hot encoding to the indices. The lambda function captures the
  # layer, so you can use them, or include them in the Keras Functional model later.
  return lambda feature: encoder(index(feature))


# ### Data preprocessing

# In[62]:


all_inputs = []
encoded_features = []


# ### Numercial preprocessing

# In[63]:


# Numerical features
for feature in list_num:
  numeric_feature = tf.keras.Input(shape=(1,), name=feature)
  normalization_layer = get_normalization_layer(feature, ds_train)
  encoded_numeric_feature = normalization_layer(numeric_feature)
  all_inputs.append(numeric_feature)
  encoded_features.append(encoded_numeric_feature)


# In[64]:


encoded_features


# ### Categorical preprocessing

# In[65]:


for feature in list_cat_int:
  categorical_feature = tf.keras.Input(shape=(1,), name=feature, dtype='int32')
  encoding_layer = get_category_encoding_layer(name=feature,
                                               dataset=ds_train,
                                               dtype='int32',
                                               max_tokens=None)
  encoded_categorical_feature = encoding_layer(categorical_feature)
  all_inputs.append(categorical_feature)
  encoded_features.append(encoded_categorical_feature)


# In[66]:


for feature in list_cat_string:
  categorical_feature = tf.keras.Input(shape=(1,), name=feature, dtype='string')
  encoding_layer = get_category_encoding_layer(name=feature,
                                               dataset=ds_train,
                                               dtype='string',
                                               max_tokens=None)
  encoded_categorical_feature = encoding_layer(categorical_feature)
  all_inputs.append(categorical_feature)
  encoded_features.append(encoded_categorical_feature)


# In[67]:


#Merge
all_features = layers.concatenate(encoded_features)


# In[68]:


all_features


# In[69]:


all_inputs


# In[70]:


# First layer
x = layers.Dense(32, activation="relu")(all_features)

# Dropout to prevent overvitting 
x = layers.Dropout(0.5)(x)

# Output layer
output = layers.Dense(1)(x)

# Group all layers 
model = tf.keras.Model(all_inputs, output)


# In[71]:


model.summary()


# In[72]:


model.compile(optimizer="adam", 
              loss ="mse", 
              metrics=["mean_absolute_error"])
              


# In[73]:


tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")


# ## Training

# In[74]:


# will stop training when there is no improvement in 3 consecutive epochs
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)


# In[75]:


model.fit(ds_train, 
            epochs=50, 
            validation_data=ds_test, 
            callbacks=[callback]
            )


# In[76]:


#Evaluate model with test-data
loss, accuracy = model.evaluate(ds_test)

print("MAE:", round(accuracy, 2))


# ## Perform inference

# In[77]:


model.save('my_car_model-mean-absolute-3')


# In[4]:


reloaded_model = tf.keras.models.load_model('my_car_model-mean-absolute-3')


# In[79]:


df.head()



# In[81]:


sample = {
    "year": 2014,
    "brand": "bmw",
    "model": "3 series",
    "type": "sedan",
    "state": "ca",
    "condition": 4.5,
    "miles": 1331.0,
    "color": "gray",
    "interior": "black",
}


# In[82]:


input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}


# In[83]:


predictions = reloaded_model.predict(input_dict)


# In[84]:


pred_list = predictions.tolist()


# In[85]:


pred_list


# In[86]:


z = (str(pred_list)[2:-2])


# In[87]:


z = float(z)


# In[88]:


print(float(round(z,2)))

