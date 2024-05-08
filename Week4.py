#!/usr/bin/env python
# coding: utf-8

# # Keras Regression Code Along Project 
# 
# Let's now apply our knowledge to a more realistic data set. Here we will also focus on feature engineering and cleaning our data!

# ## The Data
# 
# We will be using data from a Kaggle data set:
# 
# https://www.kaggle.com/harlfoxem/housesalesprediction
# 
# #### Feature Columns
#     
# * id - Unique ID for each home sold
# * date - Date of the home sale
# * price - Price of each home sold
# * bedrooms - Number of bedrooms
# * bathrooms - Number of bathrooms, where .5 accounts for a room with a toilet but no shower
# * sqft_living - Square footage of the apartments interior living space
# * sqft_lot - Square footage of the land space
# * floors - Number of floors
# * waterfront - A dummy variable for whether the apartment was overlooking the waterfront or not
# * view - An index from 0 to 4 of how good the view of the property was
# * condition - An index from 1 to 5 on the condition of the apartment,
# * grade - An index from 1 to 13, where 1-3 falls short of building construction and design, 7 has an average level of construction and design, and 11-13 have a high quality level of construction and design.
# * sqft_above - The square footage of the interior housing space that is above ground level
# * sqft_basement - The square footage of the interior housing space that is below ground level
# * yr_built - The year the house was initially built
# * yr_renovated - The year of the house’s last renovation
# * zipcode - What zipcode area the house is in
# * lat - Lattitude
# * long - Longitude
# * sqft_living15 - The square footage of interior housing living space for the nearest 15 neighbors
# * sqft_lot15 - The square footage of the land lots of the nearest 15 neighbors

# ![image.png](attachment:image.png)

# ![Latitude_and_Longitude_of_the_Earth.ng.png](attachment:Latitude_and_Longitude_of_the_Earth.ng.png)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('kc_house_data.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[ ]:





# # Exploratory Data Analysis
# 
# Perform some data analysis using the libraries above. 
# Visualise the features to understand the problem and use the appropriate features for the model. Please go ahead and do a Pandas, numpy, Matplotlib or Seaborn Tutorial if you don't understand any commands below. 

# In[5]:


df.isnull().sum()


# In[6]:


df.describe().transpose()


# ### Let's see how price columns look like i.e. how prices are distributed.

# In[7]:


plt.figure(figsize=(12,8))
sns.distplot(df['price']);


# #### Let's look at the columns - number of bedroom in more detail
# #### Write code to plot the numbers of bedrooms and the number of times they appear in the data.  
# #### The x-axis contains the number of bedrroms and the y axis will portray the number of times the particular bedroom appears in the column.
# 
# #### Hint - use sns.countplot.  

# In[8]:


sns.countplot(df['bedrooms'])


# ### Plot a scatter plot between the price and sqft_living column.

# In[9]:


plt.figure(figsize=(12,8))
sns.scatterplot(x='price',y='sqft_living',data=df)


# In[10]:


sns.boxplot(x='bedrooms',y='price',data=df)


# ### Geographical Properties

# #### Visulaize an scatter plot between price and longitude (long) and latidtude (lat) columns.

# In[11]:


plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='price',data=df)


# In[12]:


plt.figure(figsize=(12,8))
sns.scatterplot(x='price',y='long',data=df)


# In[13]:


plt.figure(figsize=(12,8))
sns.scatterplot(x='price',y='lat',data=df)


# In[14]:


plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat',data=df,hue='price')


# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png) ![image-3.png](attachment:image-3.png)

# ### Sort the values in the dataframe according to price and print first few rows.

# In[15]:


#df.sort_values?


# In[16]:


df.sort_values('price').head(20)


# In[17]:


df.sort_values('price',ascending=False).head(20)


# #### The following code visualises the price intensity with the latitude and longitude for 1% of the data.
# 
# #### You need to add comment on each line of the code.

# In[18]:


# 1% of the data

len(df)*(0.01)


# In[19]:


df.iloc[216:]


# In[20]:


# removed 1% of the most expensive houses and further sorted by decreasing price

non_top_1_perc = df.sort_values('price',ascending=False).iloc[216:]
non_top_1_perc


# In[21]:


#sns.scatterplot?


# In[22]:


# plotting map of houses sorted by prices (without 1% the most expensive houses)
plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat',
                data=non_top_1_perc,hue='price',
                palette='RdYlGn',edgecolor=None,alpha=0.2)


# ### Plotting the most expensive houses

# In[23]:


top_1_perc = df.sort_values('price',ascending=False).iloc[:216]
top_1_perc


# In[24]:


# plotting map of 1% the most expensive houses
plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat',
                data=top_1_perc,hue='price',
                palette='RdYlGn',edgecolor=None,alpha=0.2)


# In[ ]:





# ## Other Features

# #### Let's have a box plot between waterfront and price.
# #### Explain what box plot is doing?

# In[25]:


sns.boxplot(x='waterfront',y='price',data=df)


# In[ ]:





# ## Working with Feature Data

# In[26]:


df.head()


# In[27]:


df.info()


# In[28]:


# Following code is dropping the column ID. Question - why are dropping this column?

df = df.drop('id',axis=1)


# In[ ]:





# ### Feature Engineering from Date
# 
# Transform the features into useful formats to apply appropriate DNN technique!

# In[29]:


df['date'] = pd.to_datetime(df['date'])


# In[30]:


df['month'] = df['date'].apply(lambda date:date.month)


# In[31]:


df['year'] = df['date'].apply(lambda date:date.year)


# In[32]:


# check what above code is doing. 


# In[33]:


df.head()


# In[34]:


df.info()


# In[ ]:





# In[35]:


# visualize boxplot between year and price

sns.boxplot(x='year',y='price',data=df)


# In[36]:


# visualize boxplot between month and price

sns.boxplot(x='month',y='price',data=df)


# In[37]:


# approximation of maximum prices by months

df.groupby('month').mean()['price'].plot()


# In[38]:


# # approximation of maximum prices by years

df.groupby('year').mean()['price'].plot()


# In[ ]:





# In[39]:


df = df.drop('date',axis=1)


# In[ ]:





# In[40]:


df.columns


# In[41]:


# May be worth considering to remove this or feature engineer categories from it

df['zipcode'].value_counts()


# In[42]:


df = df.drop('zipcode',axis=1)


# In[43]:


df.head()


# In[44]:


df.info()


# In[45]:


# could make sense due to scaling, higher should correlate to more value

df['yr_renovated'].value_counts()


# In[46]:


df['sqft_basement'].value_counts()


# In[ ]:





# ## Scaling and Train Test Split
# 
# Scikit-Learn is used to split out the train-test library. 

# In[47]:


# First separate input and output. Input will be stored in the variable X and output in variable y.

X = df.drop('price',axis=1)
y = df['price']


# In[48]:


from sklearn.model_selection import train_test_split


# In[49]:


#train_test_split?


# In[50]:


# separate X and y into X_train, X_test, y_train, y_test i.e. getting training and testing set for the model.

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,
                            # random_state - int for reproducible output across multiple function calls
                                                    random_state=101)


# In[51]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[ ]:





# ### Scaling
# 
# Features are scaled to be in a proper range to be useful for modeling. Scaling converts all values between 0-1.

# In[52]:


from sklearn.preprocessing import MinMaxScaler


# In[53]:


scaler = MinMaxScaler()


# In[54]:


X_train= scaler.fit_transform(X_train)


# In[55]:


X_test = scaler.transform(X_test)


# In[56]:


# print shapes of X_train, X_test, y_train, y_test and see if shapes are okay.

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[ ]:





# ## Creating a Model
# 
# Build a DNN model with appropriate layers using Keras.  

# In[ ]:





# In[57]:


# TensorFlow and tf.keras

import tensorflow as tf
from tensorflow import keras


# In[58]:


print(tf.__version__)


# In[59]:


input_shape=(X_train.shape[1],)
input_shape


# In[67]:


# Develop your own Neural Network model with suitable number of input, output, and any number of hidden layers. 
# Since we are predicting a value, the number of neurons in outpit layer should be one.

model = keras.Sequential([
    keras.layers.Dense(152, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(76, activation='relu'),
    keras.layers.Dense(38, activation='relu'),
    keras.layers.Dense(19, activation='relu'),
    keras.layers.Dense(1)
])


# In[68]:


print(model.summary())


# In[ ]:





# In[ ]:


#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Activation
#from tensorflow.keras.optimizers import Adam


# In[ ]:


#model = Sequential()

#model.add(Dense(19,activation='relu'))
#model.add(Dense(19,activation='relu'))
#model.add(Dense(19,activation='relu'))
#model.add(Dense(19,activation='relu'))
#model.add(Dense(1))

#model.compile(optimizer='adam',loss='mse')


# In[ ]:





# ## Training the Model

# #### Use the following of your choice:
# 
# 1.   Optimization method
# 2.   Batch size
# 3.   Number of epochs.
# 
# Test for various optimizers and check which one performs better in terms of accuracy.
# 
# Use following APIs
# * https://keras.io/api/optimizers/
# * https://keras.io/api/models/model_training_apis/
# 

# In[69]:


model.compile(optimizer='adam', loss='mse')


# In[70]:


history = model.fit(X_train, y_train, batch_size=128, epochs=400, 
                    #validation_split=0.2, 
                    verbose=1)


# In[ ]:





# In[ ]:


#history = model.fit( x=X_train, y=y_train.values,
#                   validation_data=(X_test,y_test.values),
#                   batch_size=128,epochs=400)


# In[ ]:





# #### Following code gets the history of losses at every epoch.

# In[71]:


losses = pd.DataFrame(model.history.history)


# In[72]:


losses.plot()


# In[74]:


# Losses graphs during training

history_dict = history.history
Loss = losses
plt.figure(num=1, figsize=(15,7))
plt.plot(Loss, 'bo', label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:





# # Evaluation on Test Data
# 
# https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
# 
# Scikit-Learn has metrics to evaluate the performance.

# In[75]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score


# In[ ]:





# In[78]:


print(X_test)


# In[79]:


X_test.shape


# In[ ]:





# ### Following you will predict the output based on the input data X_test.
# 
# Lonk to API - https://keras.io/api/models/model_training_apis/#predict-method

# In[80]:


# make a predictions using (model.predict (<input_data>) method ). 
# Store the predictions in the variable predictions


# In[81]:


predictions = model.predict(X_test)


# In[ ]:





# #### Following code will test the error in the predicted values. 
# #### Error is the difference between the predictions you made and real values (y_test)

# In[83]:


print(mean_absolute_error(y_test,predictions))


# In[84]:


print(np.sqrt(mean_squared_error(y_test,predictions)))


# In[85]:


explained_variance_score(y_test,predictions)


# In[86]:


df['price'].mean()


# In[87]:


df['price'].median()


# In[88]:


# The following code plots the predicted values in a scatter plot. 
# We have also plotted the perfect predictions.


# In[89]:


# Our predictions
plt.scatter(y_test, predictions)

# Perfect predictions
plt.plot(y_test, y_test,'r')


# In[90]:


# In the following code, we have plotted the error i.e. 
# the difference between the actual and predicted values.


# In[91]:


errors = y_test.values.reshape(6484, 1) - predictions


# In[92]:


sns.distplot(errors)


# In[ ]:





# 
# ### Predicting on a brand new house
# 
# Try predicting price for a new home.

# In[96]:


single_house = df.drop('price',axis=1).iloc[0]


# In[97]:


single_house


# In[98]:


single_house.size


# In[99]:


single_house = scaler.transform(single_house.values.reshape(-1, 19))


# In[100]:


print(single_house)


# In[101]:


single_house.size


# In[102]:


model.predict(single_house)


# In[103]:


df.iloc[0]


# In[ ]:





# ### Try different optimizations like changing model architecture, activation functions, training parameters.

# In[ ]:




