#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv("D:\survey\Housing.csv")


# In[2]:


X = df.drop(['price'], axis = 1, inplace = False)
X.head()


# In[3]:


y = df.drop( [ "area" , "bedrooms" ,"bathrooms" , "stories" , "mainroad" , "guestroom" , "basement" , "hotwaterheating" , "airconditioning" , "parking" , "prefarea" , "furnishingstatus"], axis = 1 , inplace = False)
y.head()


# In[4]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[5]:


X['mainroad_n'] = le.fit_transform(X['mainroad'])
X['guestroom_n'] = le.fit_transform(X['guestroom'])
X['basement_n'] = le.fit_transform(X['basement'])
X['hotwaterheating_n'] = le.fit_transform(X['hotwaterheating'])
X['prefarea_n'] = le.fit_transform(X['prefarea'])
X['furnishinstatus_n'] = le.fit_transform(X['furnishingstatus'])
X = X.drop(['mainroad', 'guestroom' , 'basement' , 'hotwaterheating' , 'airconditioning' , 'prefarea' , 'furnishingstatus' ] , axis = 'columns')
X.head()


# In[6]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split( X , y , test_size = 0.2 , random_state = 0 )


# In[7]:


X_train


# In[8]:


y_train


# In[9]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit( X_train , y_train)


# In[10]:


X_test


# In[11]:


y_test


# In[12]:


y_pred = regressor.predict(X_test)
y_pred


# In[13]:


from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
print(score)


# In[14]:


X_test = [[5000,4,2,2,2,1,0,0,1,0,0]]
y_pred = regressor.predict(X_test)
print(y_pred)


# In[16]:


sns.distplot((y_test),bins=50);


# In[ ]:




