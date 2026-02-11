#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.neighbors import KNeighborsClassifier

def get_model():
    return KNeighborsClassifier(n_neighbors=5)

