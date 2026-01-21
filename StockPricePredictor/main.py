import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
features = ['Open', 'High', 'Low', 'Close', 'Volume']

def clean_stock_data(df):
    df['Volume'] = (df['Volume']
                .str.replace(',', '')
                .astype(float))
    return df

df = pd.read_csv(BASE_DIR / 'data' / 'TSLA-STOCK-2025-01-20-2026-01-20.csv')
df = clean_stock_data(df)

plt.subplots(figsize=(20,10))

for i, col in enumerate(features):
    plt.subplot(2, 3, i+1)
    sb.distplot(df[col])
    plt.title(f'Distribution of {col}', fontsize=15)

plt.show()