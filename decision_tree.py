#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.tree import DecisionTreeClassifier

def get_model():
    return DecisionTreeClassifier(random_state=42)

