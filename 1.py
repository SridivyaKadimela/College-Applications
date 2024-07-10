#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
sns.set(rc={'figure.figsize':(8,6)})


# In[2]:


data=pd.read_csv('college_admissions.csv')
data.drop("Serial No.",axis=1,inplace=True)
data.columns
df = data.copy()
hist = data.hist(bins=5)


# In[3]:


data.describe()


# In[4]:


data.isnull().sum()


# In[5]:


data.nunique()


# In[6]:


corr=df.corr()
mask=np.triu(np.ones_like(corr,dtype=bool))
f,ax = plt.subplots(figsize=(8,6))
cmap = sns.diverging_palette(500000,42069, as_cmap=True)
sns.heatmap(corr,cmap=cmap,mask=mask,linewidth=0.5,square=True,center=0)


# In[7]:


plt.figure(figsize=(15,10))

sns.heatmap(data.corr(),annot=True,cmap='viridis')


# In[8]:


sns.pairplot(data);


# In[9]:


corr


# ### You can clearly see the correlation between Chance of Admit and GRE, TOEFL and CGPA score

# In[10]:


newdata = data[['CGPA','GRE Score','TOEFL Score','University Rating']]
newdata2 = data[['CGPA','University Rating']]
newdata3 = data[['GRE Score','TOEFL Score']]


# In[11]:


newdata2.boxplot(return_type='axes')


# In[12]:


newdata3.boxplot(return_type='axes')


# ## As you can see, GRE, TOEFL and CGPA are the top three contributors to the predictions

# #### There seem to be no outliers or irregularities in data

# # Linear Regression

# In[14]:


train_errorsL = []
test_errorsL = []
X = data.copy();
X.drop('Chance of Admit ',axis = 1,inplace = True)
y = data['Chance of Admit ']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
# ss = StandardScaler()
# X_train = ss.fit_transform(X_train)
# X_test = ss.transform(X_test)
model = LinearRegression()
accuracy = model.fit(X_train, y_train).score(X_train, y_train)
train_errorsL.append(1-accuracy)
print('Training Accuracy: ',accuracy)
print('Training error: ',1-accuracy)
y_predlr = model.predict(X_test)
mse = mean_squared_error(y_predlr,y_test)
test_errorsL.append(mse)
print('Test MSE: ',mse)


# In[15]:


plt.scatter(y_test,y_predlr)
plt.show()


# ##### The task asks for a decision boundary. I dont understand how a decision boundary can be drawn for a regression model

# # PCA of 2 dimensions

# In[16]:


pca = PCA(n_components = 2)
X2D = pca.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X2D, y, test_size=0.2,random_state=0)
model2 = LinearRegression()
accuracy = model2.fit(X_train, y_train).score(X_train, y_train)

train_errorsL.append(1-accuracy)
print('Training Accuracy: ',accuracy)
print('Training error: ',1-accuracy)
y_pred = model2.predict(X_test)
mse = mean_squared_error(y_pred,y_test)
print('Test MSE: ',mse)
test_errorsL.append(mse)


# In[17]:


plt.scatter(y_test,y_pred)
plt.show()


# In[18]:


final_preds = pd.DataFrame(
{'LinearAllfeatures': y_predlr,
  'PCA2D':y_pred  
}
)
final_preds.to_csv('preds.csv')


# In[19]:


model_list = ['AllfeaturesLR','PCA2D']
plt.figure(figsize=(20, 10))
plt.bar(model_list, train_errorsL)
plt.xlabel('Models', fontsize=20)
plt.ylabel('Training error', fontsize=20, rotation=0)
plt.show()


# In[20]:


model_list = ['AllfeaturesLR','PCA2D']
plt.figure(figsize=(20, 10))
plt.bar(model_list, test_errorsL)
plt.xlabel('Models', fontsize=20)
plt.ylabel('Test error(MSE)', fontsize=20, rotation=0)
plt.show()


# # The model with all the features does perform a little better. But it would not be wise to do that with data with thousands of dimensions. As the performance is more or less similar with lesser dimensions, it is better to use PCA to reduce the dimensions of the data so that it is easier for the machine to detect patterns

# In[ ]:




