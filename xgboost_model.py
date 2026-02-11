#!/usr/bin/env python
# coding: utf-8

# In[1]:


from xgboost import XGBClassifier

def get_model():
    return XGBClassifier(eval_metric="logloss")

