#import libraries
import xgboost as xg
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib as plt
from scipy.stats import norm
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold, KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from sklearn.metrics import r2_score
import data_ML as ML

np.set_printoptions(suppress=True)

df = ML.getdata(False)

def modifydata(df):
    data = []
    df_calls = df[df.optionType == 1].drop(columns='optionType')
    df_american_calls = df_calls[df_calls.method == 1].drop(columns='method')
    # df_american_calls = df_american_calls.drop(columns='index')
    df_american_calls = df_american_calls[df_american_calls.lastPrice > 5]
    # df_american_calls = df_american_calls.drop(columns='index')
    y = np.asarray(df_american_calls[['lastPrice']])
    df_american_calls = df_american_calls.drop(columns='lastPrice')

    for line in range(len(df_american_calls)):
        data_tmp = df_american_calls.iloc[line]
        data.append(data_tmp)
    data = np.asarray(data)

    return data, y

# df_train = ML.getdata(True)


X, y = modifydata(df)
# df_train, y_train, full_data2 = modifydata(df_train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

XGBr = xg.XGBRegressor(learning_rate=0.1, n_estimators=900)
cvs_XGBr = cross_val_score(XGBr, X_train, y_train, scoring='neg_root_mean_squared_error', cv=KFold(n_splits=10))
XGBr_score = cvs_XGBr
print(XGBr_score)

XGBr.fit(X_train, y_train)

def xgb_prediction(S_0, r, T, K , sigma):
    data = np.asarray(S_0, r, T, K, sigma)
    prediction = XGBr.predict(data)
    return prediction