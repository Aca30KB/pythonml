import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

df = pd.read_csv('freMTPL2freq.csv')

df.isnull().sum()

df = df.drop('IDpol', axis=1)

df.describe()

df = pd.get_dummies(df, columns=['Area', 'VehBrand', 'VehGas', 'Region'], drop_first=True)
df['ClaimFreq'] = df['ClaimNb'] / df['Exposure']
df.replace([float('inf')], 0, inplace=True)

df.describe()

df

X = df.drop(['ClaimNb', 'Exposure', 'ClaimFreq'], axis=1)
y = df['ClaimFreq']

sample_weights = df['Exposure']

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, sample_weights, test_size=0.2, random_state=42, shuffle=True)

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100,
                         learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train, sample_weight=w_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred, sample_weight=w_test)
print(f"Mean squared error (weighted by exposure):{mse:.4f}")

''' k fold cross validation'''

pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100,
                         learning_rate=0.1, max_depth=5, random_state=42))
pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)

kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
scores = []

for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist: %s, Acc: %.3f' % (k+1, np.bincount(y_train[train]), score))
