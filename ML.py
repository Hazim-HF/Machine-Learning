# ===== Linear Regression =====
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import os
os.chdir('D:/01. Education/02. Master/Semester 2/Machine-Learning')

boston = pd.read_csv('Data/boston.csv')

y = boston['medv']
x = boston.drop('medv', axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
scale = StandardScaler()
ohe = OneHotEncoder(sparse_output = False)

x_train_num = scale.fit_transform(x_train.select_dtypes(include=['float64', 'int64']))
x_test_num = scale.fit_transform(x_test.select_dtypes(include=['float64', 'int64']))

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

model = lr.fit(x_train_num, y_train)

prediction = model.predict(x_test_num)

model.coef_

model.intercept_

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, prediction)
r2 = r2_score(y_test, prediction)

# ===== Logistic Regression =====
import pandas as pd
default = pd.read_csv('data/default.csv')
default.dtypes

y = default['default']
x = default.drop('default', axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
scale = StandardScaler()
ohe = OneHotEncoder(sparse_output=False) # optional: drop='first'
x_train_num = scale.fit_transform(x_train.select_dtypes(include=['float64', 'int64']))
x_test_num = scale.fit_transform(x_test.select_dtypes(include=['float64', 'int64']))
x_train_cat = ohe.fit_transform(x_train.select_dtypes(include=['object']))
x_test_cat = ohe.fit_transform(x_test.select_dtypes(include=['object']))

x_train_processed = np.hstack((x_train_num, x_train_cat))
x_test_processed = np.hstack((x_test_num, x_test_cat))

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

log_reg_model = log_reg.fit(x_train_processed, y_train)

y_predict = log_reg_model.predict(x_test_processed)

from sklearn.metrics import (accuracy_score, recall_score, 
                             precision_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report)

acc = accuracy_score(y_test, y_predict)
print(confusion_matrix(y_test, y_predict))
rec = recall_score(y_test, y_predict)
prec = precision_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)
auc = roc_auc_score(y_test, y_predict)
print(classification_report(y_test, y_predict))

# ===== Visualization =====

import seaborn as sns

sns.scatterplot(x=y_test, y=prediction)

# ===== Cross Validation =====

# ===== Principal Component Analysis (PCA) =====
from sklearn.decomposition import PCA

pca = PCA()

x_reduces = pca.fit_transform(x_train_num)

len(x_reduces[0])

len(x_train_num)

# ===== Linear Discriminant Analysis (LDA) =====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Define column names
cls = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read the data set
dataset = pd.read_csv(url, names=cls)

# Divide the data set into features (X) and target variable (y)
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data set into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Apply Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
