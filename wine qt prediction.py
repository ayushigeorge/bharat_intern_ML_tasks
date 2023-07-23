#!/usr/bin/env python
# coding: utf-8

# # BHARAT INTERN :MACHINE LEARNING INTERNSHIP

# # TASK 3: WINE QUALITY PREDICTION

# # BY AYUSHI GEORGE

# In[1]:


# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ### Load the dataset (assuming you have a CSV file with wine quality data)

# In[2]:


data = pd.read_csv('WineQT.csv')

print(data.head())


# In[3]:


data.info()


# In[4]:


data.describe().T


# In[5]:


data.isnull().sum()


# ### Let’s draw the histogram to visualise the distribution of the data with continuous values in the columns of the dataset

# In[6]:


data.hist(bins=20, figsize=(10, 10))
plt.show()


# ### let’s draw the count plot to visualise the number data for each quality of wine 

# In[7]:


plt.bar(data['quality'], data['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()


# ### Heat Map

# In[8]:


plt.figure(figsize=(12, 12))
sb.heatmap(data.corr() > 0.7, annot=True, cbar=False)
plt.show()


# ### Assuming 'fixed acidity', 'volatile acidity', 'citric acid', and 'alcohol' as features

# In[9]:


X = data[['fixed acidity', 'volatile acidity', 'citric acid', 'alcohol']].values
y = data['quality'].values


# ### Split the data into training and testing sets

# In[10]:



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### Create and train the Logistic Regression model

# In[11]:



model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# ### Make predictions on the test set

# In[12]:



y_pred = model.predict(X_test)


# ### Evaluate the model 

# In[13]:



accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# ### GENERATING OUTPUT
# 

# In[14]:


print("Classification Report:")
print(classification_report(y_test, y_pred))


# ### CONFUSION MATRIX

# In[15]:



conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# # Thank you

# In[ ]:




