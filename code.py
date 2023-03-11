#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np #linear algebra
import pandas as pd #data processing, csv file (pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns #plotting graphs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from pandas.plotting import scatter_matrix

df=pd.read_csv("universal.csv")
df


# In[19]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


plt.scatter(df['attacktype'],df['group_name'])


# In[21]:


plt.scatter(df['targtype1_txt'],df['group_name'])


# In[22]:


plt.scatter(df['city'],df['group_name'])


# In[23]:


X = df[['attacktype','targtype1_txt' , 'city']]
y = df['group_name']


# In[26]:


X


# In[27]:


y


# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


print(df.describe())


# In[31]:


array = dataset.values
X = array[:,0:3]
Y = array[:,3]
validation_size = 0.20
seed = 6
X_train, X_test, Y_train, Y_test = model_classificatoion


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[57]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# In[58]:


len(X_train)


# In[59]:


len(X_test)


# In[1]:


df.hist()
plot.show()


# In[2]:


from sklearn.linear_model import LinearRegression
clf = LinearRegression()


# In[66]:


clf.fit(X_train, y_train)


# In[63]:


clf.predict(X_test)


# In[64]:


y_test


# In[51]:


clf.score(X_test, y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


f1 = pd.get_dummies(df.attacktype)
f1


# In[6]:


f2 = pd.get_dummies(df.group_name)
f2


# In[7]:


f3 = pd.get_dummies(df.targtype1_txt)
f3


# In[10]:


cc1 = pd.concat([df, f1], axis=1)
cc1


# In[11]:


cc2 = pd.concat([cc1, f2], axis=1)
cc2


# In[12]:


cc3 = pd.concat([cc2, f3], axis=1)
cc3


# In[13]:


cc4 = pd.concat([f3])


# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


plt.scatter(df['attacktype'],df['group_name'])


# In[ ]:




