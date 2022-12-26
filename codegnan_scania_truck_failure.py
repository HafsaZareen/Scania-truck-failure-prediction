#!/usr/bin/env python
# coding: utf-8

# In[204]:
import pickle
import sklearn
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


import warnings
warnings.filterwarnings(action='ignore')


# In[205]:


# test_df = pd.read_csv("G:\\94_65\\archive\\aps_failure_test_set.csv",nrows=1000)
train_df = pd.read_csv('G:\\94_65\\archive\\aps_failure_training_set.csv',nrows=1000)


# In[206]:


train_df= train_df.replace('neg',0)
train_df= train_df.replace('pos',1)


# In[207]:


# train_df=train_df.fillna(0,inplace="na")
train_df= train_df.replace('na',0)
# test_df= train_df.replace('na',0)



# In[208]:


train_df


# In[209]:


std = StandardScaler()


# In[210]:


X = train_df.drop(columns='class', axis=1)
Y = train_df['class']


# In[211]:


print(X)


# In[212]:


print(Y)


# In[213]:


new_df = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)


# In[214]:


print(new_df)


# In[215]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=2)


# In[216]:


model = LogisticRegression()


# In[217]:


lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(Y_train)


# In[218]:


model.fit(X_train ,y_transformed)


# In[219]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,y_transformed)


# In[220]:


print(training_data_accuracy*100)


# In[238]:


n_components = 5

pca = PCA(n_components=n_components)
pca.fit(X_train)

X_train_reduced = pd.DataFrame(pca.transform(X_train), index=X_train.index, columns=["PC" + str(i + 1) for i in range(n_components)])
X_test_reduced = pd.DataFrame(pca.transform(X_test), index=X_test.index, columns=["PC" + str(i + 1) for i in range(n_components)])


# In[239]:


X_train_reduced


# In[240]:


X_test_reduced


# In[241]:


plt.figure(figsize=(16, 10))
sns.barplot(x=pca.explained_variance_ratio_, y=["PC" + str(i + 1) for i in range(n_components)], orient='h', palette='husl')
plt.show()


# In[242]:


reduced_model = LogisticRegression()


# In[243]:


reduced_model.fit(X_train_reduced ,y_transformed)


# In[244]:


# accuracy on training data
X_train_red_prediction = reduced_model.predict(X_train_reduced)
training_data_accuracy1 = accuracy_score(X_train_red_prediction,y_transformed)


# In[245]:


print(training_data_accuracy1*100)


# In[247]:


Y_train


# In[248]:


Y_test



pickle.dump(reduced_model,open("model.pkl","wb"))




