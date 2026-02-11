#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import RandomForestClassifier

def get_model():
    return RandomForestClassifier(n_estimators=100, random_state=42)

