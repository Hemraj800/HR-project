#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt


# In[8]:


df_tr=pd.read_csv(r"C:\Users\HP\Downloads\Hr file.csv.zip")


# In[11]:


df_tr


# In[12]:


df_tr.shape


# In[13]:


df_tr.isnull()


# In[14]:


df_tr.isnull().sum()


# In[15]:


sns.heatmap(df_tr.isnull())
plt.show()


# In[16]:


df_tr.isna().sum()


# In[17]:


df_tr.columns


# In[18]:


df_tr.info()


# In[19]:


df_tr.dtypes


# In[20]:


df_tr.drop(['Age'],axis=1,inplace=True)
df_tr.head()


# In[21]:


df_tr.columns


# In[22]:


df_tr['Attrition'].unique()


# In[23]:


df_tr['BusinessTravel'].unique()


# In[24]:


df_tr['DailyRate'].unique()


# In[25]:


df_tr['Department'].unique()


# In[26]:


df_tr['DistanceFromHome'].unique()


# In[27]:


sns.histplot('Department')
plt.show()


# In[28]:


from sklearn.preprocessing import LabelEncoder # import


# In[29]:


le=LabelEncoder()
for i in df_tr.drop(['Department'],axis=1):
    df_tr[i]=le.fit_transform(df_tr[i])
df_tr


# In[31]:


df_tr.head()


# In[32]:


df_tr.dtypes


# In[36]:


df_tr.corr()


# In[37]:


plt.figure(figsize=(15,8))
sns.heatmap(df_tr.corr(),annot=True)


# In[ ]:




