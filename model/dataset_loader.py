#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import requests
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from urllib.request import urlopen
from io import StringIO


def load_credit_card_default():
    """
    Loads UCI Credit Card Default dataset,
    scales features and returns train-test split.
    """

    # Dataset URL (UCI repository)
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
        "default%20of%20credit%20card%20clients.xls"
    )

    # Download dataset
    response = requests.get(url)
    response.raise_for_status()

    # Read Excel file (skip first descriptive row)
    df = pd.read_excel(BytesIO(response.content), header=1)

    # Rename target column
    df = df.rename(columns={"default payment next month": "target"})

    # Drop ID column (not a feature)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Split features and target
    X = df.drop("target", axis=1)
    feature_columns = X.columns
    y = df["target"]

    # Scale features (required for LR, KNN, NB)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler, feature_columns

