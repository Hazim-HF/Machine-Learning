import os
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# upload data
os.chdir('D:/01. Education/02. Master/Semester 2/Machine-Learning')
hits = pd.read_csv('data/hitters.csv')

# Data Preprocessing
hits.isna().sum().sum()