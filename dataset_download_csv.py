#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import requests
from io import BytesIO

def load_credit_card_default():
    """
    Sklearn-like loader for the UCI Credit Card Default dataset.
    Returns a dictionary-style object similar to sklearn.datasets loaders.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"

    # Download dataset
    response = requests.get(url)
    response.raise_for_status()

    # Load Excel file (skip description row)
    df = pd.read_excel(BytesIO(response.content), header=1)

    # Rename target column
    df = df.rename(columns={"default payment next month": "target"})

    # Separate features and target
    X = df.drop(columns=["target"])
    y = df["target"]

    return {
        "data": X.values,
        "target": y.values,
        "feature_names": X.columns.tolist(),
        "target_names": ["no_default", "default"],
        "frame": df,
        "DESCR": "UCI Credit Card Default Dataset"
    }

def save_dataset_to_csv():
    # Load dataset (sklearn-style)
    data = load_credit_card_default()

    # Convert to DataFrame (same pattern as breast cancer code)
    df = pd.DataFrame(data["data"], columns=data["feature_names"])
    df["target"] = data["target"]

    # Save to CSV
    file_name = "credit_card_default_dataset.csv"
    df.to_csv(file_name, index=False)

    print(f"Dataset successfully saved as: {file_name}")
    print(f"Total Rows: {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}")

if __name__ == "__main__":
    save_dataset_to_csv()

