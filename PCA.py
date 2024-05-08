#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Practical 8 : To write a program for Dimensionality Reduction.

import pandas as pd
from sklearn.datasets import load_digits


# In[3]:


dataset = load_digits()
dataset.keys()


# In[7]:


dataset.data.shape


# In[8]:


dataset.data[0].reshape(8, 8)


# In[12]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.gray()
plt.matshow(dataset.data[50].reshape(8, 8))


# In[13]:


dataset.target


# In[16]:


import numpy as np

np.unique(dataset.target)


# In[18]:


dataset.target[5]


# In[21]:


pd.DataFrame(dataset.data)


# In[27]:


df = pd.DataFrame(dataset.data, columns = dataset.feature_names)
df.head()


# In[28]:


df.describe()


# In[30]:


x = df
y = dataset.target


# In[31]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_scaled


# In[44]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.2, random_state = 30)


# In[45]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)
model.score(x_test, y_test)


# In[47]:


from sklearn.decomposition import PCA

pca = PCA(0.95)

x_pca = pca.fit_transform(x)
x_pca.shape


# In[48]:


x.shape


# In[49]:


pca.explained_variance_ratio_


# In[58]:


pca.n_components_


# In[59]:


x_train_pca, x_test_pca, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state = 30)


# In[64]:


model = LogisticRegression(max_iter = 1000)

model.fit(x_train_pca, y_train)
model.score(x_test_pca, y_test)


# In[65]:


pca = PCA(n_components = 2)
x_pca = pca.fit_transform(x)
x_pca.shape


# In[66]:


x_pca


# In[67]:


pca.explained_variance_ratio_


# In[71]:


X_train_pca, X_test_pca, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=30)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_pca, y_train)
model.score(X_test_pca, y_test)


# In[72]:


pca.n_components_

