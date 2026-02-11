#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.linear_model import LogisticRegression

def get_model():
    return LogisticRegression(max_iter=500)

