#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


car=pd.read_csv(r'C:/Users/shank/Desktop/CARS.csv')


# In[4]:


car.head()


# In[5]:


car.info()


# In[6]:


car.dtypes


# In[7]:


#Remove irrelvant features from the dataset
car = car.drop(['Model','DriveTrain','Invoice', 'Origin', 'Type'], axis=1)
car.head(5)


# In[8]:


car.shape


# In[9]:


car = car.drop_duplicates(subset='MSRP', keep='first')
car.count()


# In[10]:


#find null values in the dataset
print(car.isnull().sum())


# In[11]:


#find the index number of the null values 
car[car['Cylinders'].isnull()].index.tolist()


# In[12]:


#assign the mean value of cylinders to null value locations i.e 247 and 248
newval=car['Cylinders'].mean()
car['Cylinders'][247]=round(newval)
newval=car['Cylinders'].mean()
car['Cylinders'][248]=round(newval)


# In[13]:


car['Cylinders'][248]


# In[14]:


#now we see confirm that are no more null values in the dataset
print(car.isnull().sum())


# In[15]:


car['MSRP'] = car['MSRP'].str.replace(',', '')


# In[16]:


car['MSRP'] = car['MSRP'].str.replace('$', '')


# In[17]:


car.head()


# In[18]:


#convert object to integer datatype
car['MSRP']=pd.to_numeric(car['MSRP'],errors='coerce')


# In[19]:


car.dtypes


# In[20]:


sns.boxplot(x=car['MSRP'])


# In[21]:


Q1 = car.quantile(0.25)
Q3 = car.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[22]:


car = car[~((car < (Q1-1.5 * IQR)) |(car> (Q3 + 1.5 * IQR))).any(axis=1)]


# In[23]:


sns.boxplot(car['MSRP'])


# In[24]:


car.corr()


# In[25]:


plt.figure(figsize=(10,3))
c= car.corr()
sns.heatmap(c,cmap="BuGn_r",annot=True)


# In[26]:


plt.scatter(car['Horsepower'], car['MSRP'])
plt.xlabel('Horsepower')
plt.ylabel('MSRP');


# In[27]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[30]:


las_reg = linear_model.Lasso(alpha=0.1)


# In[52]:


X=car.drop(['MSRP','Make'],axis=1)
Y=car['MSRP']


# In[53]:


X = X.to_numpy()
X.ndim


# In[54]:


Y = Y.to_numpy()
Y.ndim


# In[77]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=40)


# In[82]:


#lasso Regression
las_reg.fit(X_train, Y_train)
pred = las_reg.predict(X_test)
print("Mean Absolute Error is :", mean_absolute_error(Y_test, pred))
print("Mean Squared Error is :", mean_squared_error(Y_test, pred))
print("Coeffients are : ", las_reg.coef_)
print("Intercepts are :" ,las_reg.intercept_)
print("The R2 square value of Lasso is :", r2_score(Y_test, pred)*100)


# In[83]:


pred[1:5]


# In[84]:


#Random Forest
Random_model = RandomForestRegressor()
Random_model.fit(X_train, Y_train)

pred = Random_model.predict(X_test)
print("Mean Absolute Error is :", mean_absolute_error(Y_test, pred))
print("Mean Squared Error is :", mean_squared_error(Y_test, pred))
print("The R2 square value of Random Forest is :", r2_score(Y_test, pred)*100)


# In[85]:


#Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)

pred = linear_model.predict(X_test)
print("Mean Absolute Error is :", mean_absolute_error(Y_test, pred))
print("Mean Squared Error is :", mean_squared_error(Y_test, pred))
print("Coeffients are : ", linear_model.coef_)
print("Intercepts are :" ,linear_model.intercept_)
print("The R2 square value of Lasso is :", r2_score(Y_test, pred)*100)


# In[ ]:




