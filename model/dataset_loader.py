import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import BytesIO
import pandas as pd

def load_credit_card_default():

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"

    # Create session with retry strategy
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to download dataset from UCI: {e}")

    df = pd.read_excel(BytesIO(response.content), header=1)

    # Rename target
    df = df.rename(columns={"default payment next month": "target"})

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    X = df.drop("target", axis=1)
    feature_columns = X.columns
    y = df["target"]

    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler, feature_columns
