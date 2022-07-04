#!/usr/bin/env python
# coding: utf-8

# # TF-try

# # Load packages

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


# In[2]:


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)


# # Import Dataset

# In[3]:


raw_dataset = pd.read_csv("car_prices.csv", on_bad_lines="skip")


# In[5]:


df = raw_dataset.copy()
df.head(2)


# ## Get an overview

# In[6]:


print(f"We have {len(df.index):,} observations and {len(df.columns)} columns in our dataset.")


# In[7]:


df.columns


# In[9]:


df.info()


# # Clean the data

# ## Remove unnessecary columns
# 
# * trim
# * vin
# * mmr

# In[10]:


df = df.drop(["trim", "vin", "mmr"], axis=1)


# In[11]:


df.info()


# ## Rename columns

# In[12]:


df = df.rename(columns={
"make" : "brand",
"body" : "type",
"transmission" : "drivetrain",
"odometer" : "miles"} 
    )


# ## Remove missing values
# 
# 

# In[13]:


# Check for missing values
df.isna().sum()


# In[14]:


# Drop missing values
df = df.dropna()


# In[15]:


# Check missing values again
df.isna().sum()


# ## Transform in lowercase

# In[16]:


df["brand"] = df["brand"].str.lower()
df["model"] = df["model"].str.lower()
df["type"] = df["type"].str.lower()


# In[17]:


df["brand"].head()


# # Categorical or numeric?
# 
# * year = categorial
# * brand = categorial
# * model = categorial
# * type = categorial
# * drivetrain = categorial
# * state = categorial
# * condition = categorial
# * miles = numeric
# * color = categorial
# * interior = categorial
# * seller = categorial
# * ratingprice = numeric
# * sellingprice = numeric
# * saledate = categorial

# In[23]:


# Transform into categorical

for cat in ["year", "brand", "model", "type", "drivetrain", "state", "condition", "color", "interior", "seller", "saledate"]:
    df[cat] = df[cat].astype("category")


df.dtypes


# In[88]:


df["sellingprice"] = df["sellingprice"].astype(float)


# ## Prepare the data
# 
# 

# In[70]:


# list of all numerical data
list_num = df.select_dtypes(include=[np.number]).columns.tolist()

# list of all categorical data
list_cat = df.select_dtypes(include=['category']).columns.tolist()

print(list_num, list_cat)


# ### Split the data into training and test sets
# 
# Now, split the dataset into a training set and a test set. You will use the test set in the final evaluation of your models.

# In[71]:


df_train = df.sample(frac=0.8, random_state=42)
df_test = df.drop(df_train.index)


# ### Inspect the data
# 
# Review the joint distribution of a few pairs of columns from the training set.
# 
# 

# In[72]:


sns.pairplot(df_train[['sellingprice', 'miles']], diag_kind='kde')


# Let's also check the overall statistics. Note how each feature covers a very different range:

# In[73]:


df_train.describe().transpose()


# ### Split features from labels
# 
# Separate the target value—the "label"—from the features. This label is the value that you will train the model to predict.

# In[91]:


train_features = df_train.copy()
test_features = df_test.copy()

train_labels = train_features.pop('sellingprice')
test_labels = test_features.pop('sellingprice')


# ## Normalization
# 
# In the table of statistics it's easy to see how different the ranges of each feature are:

# In[93]:


#train_features =train_features.to_numpy()
train_features = tf.convert_to_tensor(train_features)

print(type(train_features))


# In[94]:


train_features


# In[40]:


df_train.describe().transpose()[['mean', 'std']]


# It is good practice to normalize features that use different scales and ranges.
# 
# One reason this is important is because the features are multiplied by the model weights. So, the scale of the outputs and the scale of the gradients are affected by the scale of the inputs.
# 
# Although a model *might* converge without feature normalization, normalization makes training much more stable.
# 
# Note: There is no advantage to normalizing the one-hot features—it is done here for simplicity. For more details on how to use the preprocessing layers, refer to the [Working with preprocessing layers](https://www.tensorflow.org/guide/keras/preprocessing_layers) guide and the [Classify structured data using Keras preprocessing layers](../structured_data/preprocessing_layers.ipynb) tutorial.

# ### The Normalization layer
# 
# The `tf.keras.layers.Normalization` is a clean and simple way to add feature normalization into your model.
# 
# The first step is to create the layer:

# In[41]:


normalizer = tf.keras.layers.Normalization(axis=-1)


# Then, fit the state of the preprocessing layer to the data by calling `Normalization.adapt`:

# In[50]:


tf.dtypes.DType()


# In[42]:


normalizer.adapt(np.array(train_features))


# Calculate the mean and variance, and store them in the layer:

# In[43]:


print(normalizer.mean.numpy())


# When the layer is called, it returns the input data, with each feature independently normalized:

# In[82]:


first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())


# ### Linear regression with multiple inputs

# You can use an almost identical setup to make predictions based on multiple inputs. This model still does the same $y = mx+b$ except that $m$ is a matrix and $b$ is a vector.
# 
# Create a two-step Keras Sequential model again with the first layer being `normalizer` (`tf.keras.layers.Normalization(axis=-1)`) you defined earlier and adapted to the whole dataset:

# In[77]:


linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])


# When you call `Model.predict` on a batch of inputs, it produces `units=1` outputs for each example:

# In[78]:


linear_model.predict(train_features[:10])


# When you call the model, its weight matrices will be built—check that the `kernel` weights (the $m$ in $y=mx+b$) have a shape of `(9, 1)`:

# In[79]:


linear_model.layers[1].kernel


# Configure the model with Keras `Model.compile` and train with `Model.fit` for 100 epochs:

# In[80]:


linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


# In[81]:


get_ipython().run_cell_magic('time', '', 'history = linear_model.fit(\n    train_features,\n    train_labels,\n    epochs=100,\n    # Suppress logging.\n    verbose=0,\n    # Calculate validation results on 20% of the training data.\n    validation_split = 0.2)\n')


# Using all the inputs in this regression model achieves a much lower training and validation error than the `horsepower_model`, which had one input:

# In[38]:


plot_loss(history)


# Collect the results on the test set for later:

# In[39]:


test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=0)


# ## Regression with a deep neural network (DNN)

# In the previous section, you implemented two linear models for single and multiple inputs.
# 
# Here, you will implement single-input and multiple-input DNN models.
# 
# The code is basically the same except the model is expanded to include some "hidden" non-linear layers. The name "hidden" here just means not directly connected to the inputs or outputs.

# These models will contain a few more layers than the linear model:
# 
# * The normalization layer, as before (with `horsepower_normalizer` for a single-input model and `normalizer` for a multiple-input model).
# * Two hidden, non-linear, `Dense` layers with the ReLU (`relu`) activation function nonlinearity.
# * A linear `Dense` single-output layer.
# 
# Both models will use the same training procedure so the `compile` method is included in the `build_and_compile_model` function below.

# In[40]:


def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model


# ### Regression using a DNN and a single input

# Create a DNN model with only `'Horsepower'` as input and `horsepower_normalizer` (defined earlier) as the normalization layer:

# In[41]:


dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)


# This model has quite a few more trainable parameters than the linear models:

# In[42]:


dnn_horsepower_model.summary()


# Train the model with Keras `Model.fit`:

# In[43]:


get_ipython().run_cell_magic('time', '', "history = dnn_horsepower_model.fit(\n    train_features['Horsepower'],\n    train_labels,\n    validation_split=0.2,\n    verbose=0, epochs=100)\n")


# This model does slightly better than the linear single-input `horsepower_model`:

# In[44]:


plot_loss(history)


# If you plot the predictions as a function of `'Horsepower'`, you should notice how this model takes advantage of the nonlinearity provided by the hidden layers:

# In[45]:


x = tf.linspace(0.0, 250, 251)
y = dnn_horsepower_model.predict(x)


# In[46]:


plot_horsepower(x, y)


# Collect the results on the test set for later:

# In[47]:


test_results['dnn_horsepower_model'] = dnn_horsepower_model.evaluate(
    test_features['Horsepower'], test_labels,
    verbose=0)


# ### Regression using a DNN and multiple inputs

# Repeat the previous process using all the inputs. The model's performance slightly improves on the validation dataset.

# In[48]:


dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()


# In[49]:


get_ipython().run_cell_magic('time', '', 'history = dnn_model.fit(\n    train_features,\n    train_labels,\n    validation_split=0.2,\n    verbose=0, epochs=100)\n')


# In[50]:


plot_loss(history)


# Collect the results on the test set:

# In[51]:


test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)


# ## Performance

# Since all models have been trained, you can review their test set performance:

# In[52]:


pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T


# These results match the validation error observed during training.

# ### Make predictions
# 
# You can now make predictions with the `dnn_model` on the test set using Keras `Model.predict` and review the loss:

# In[53]:


test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# It appears that the model predicts reasonably well.
# 
# Now, check the error distribution:

# In[54]:


error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')


# If you're happy with the model, save it for later use with `Model.save`:

# In[55]:


dnn_model.save('dnn_model')


# If you reload the model, it gives identical output:

# In[56]:


reloaded = tf.keras.models.load_model('dnn_model')

test_results['reloaded'] = reloaded.evaluate(
    test_features, test_labels, verbose=0)


# In[57]:


pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T


# ## Conclusion
# 
# This notebook introduced a few techniques to handle a regression problem. Here are a few more tips that may help:
# 
# - Mean squared error (MSE) (`tf.keras.losses.MeanSquaredError`) and mean absolute error (MAE) (`tf.keras.losses.MeanAbsoluteError`) are common loss functions used for regression problems. MAE is less sensitive to outliers. Different loss functions are used for classification problems.
# - Similarly, evaluation metrics used for regression differ from classification.
# - When numeric input data features have values with different ranges, each feature should be scaled independently to the same range.
# - Overfitting is a common problem for DNN models, though it wasn't a problem for this tutorial. Visit the [Overfit and underfit](overfit_and_underfit.ipynb) tutorial for more help with this.
