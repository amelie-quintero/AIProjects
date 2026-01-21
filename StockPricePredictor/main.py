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
df = pd.read_csv(BASE_DIR / 'data' / 'TSLA-STOCK-2025-01-20-2026-01-20.csv')

print(df.head())