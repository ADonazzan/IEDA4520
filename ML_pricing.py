from keras.models import load_model
import numpy as np
# from keras.utils import Sequence
# from keras.models import Sequential
import pickle

# Load the saved XGBoost model from a file
#American files
filename1, filename2, filename3, filename4 = './Saved_ML_models/XGBr_model_calls.sav', './Saved_ML_models/XGBr_model_puts.sav', './Saved_ML_models/DTR_model_puts.sav', './Saved_ML_models/DTR_model_calls.sav'

#EU files
eu_file1, eu_file2, eu_file3, eu_file4 = "./Saved_ML_models/XGBr_eu_calls.sav", "./Saved_ML_models/XGBr_eu_puts.sav", './Saved_ML_models/DTR_eu_calls.sav', './Saved_ML_models/DTR_eu_puts.sav'

# filename1, filename2 = './Saved_ML_models/XGBr_model_calls.sav', './Saved_ML_models/XGBr_model_puts.sav'

#XGBr
XGBr_model_am_calls = pickle.load(open(filename1, 'rb'))
XGBr_model_am_puts = pickle.load(open(filename2, 'rb'))
XGBr_model_eu_calls = pickle.load(open(eu_file1, 'rb'))
XGBr_model_eu_puts = pickle.load(open(eu_file2, 'rb'))

#DTR
DTR_model_am_p = pickle.load(open(filename3, 'rb'))
DTR_model_am_c = pickle.load(open(filename4, 'rb'))
DTR_model_eu_c = pickle.load(open(eu_file3, 'rb'))
DTR_model_eu_p = pickle.load(open(eu_file4, 'rb'))


'''
---------------------------------------------------
Xtreme Gradient Boost Algorithm American Calls/Puts
---------------------------------------------------
'''

def XGBr_am_calls(S0, K, T, sigma, r):
    X_new = np.asarray([[K, T, S0, sigma, r]])
    y_pred = XGBr_model_am_calls.predict(X_new)[0]
    return y_pred

# print(XGBr_am_calls(230, 246/365, 189.71, 0.3318, 0.269))

def XGBr_am_puts(S0, K, T, sigma, r):
    X_new = np.asarray([[K, T, S0, sigma, r]])
    y_pred = XGBr_model_am_puts.predict(X_new)[0]
    return y_pred

# print(XGBr_am_puts(230, 246/365, 189.71, 0.3318, 0.269))

def XGBr_eu_calls(S0, K, T, sigma, r):  
    X_new = np.asarray([[K, T, S0, sigma, r]])
    y_pred = XGBr_model_eu_calls.predict(X_new)
    return y_pred[0]

# print(XGBr_eu_calls(230, 246/365, 189.71, 0.3318, 0.269))

def XGBr_eu_puts(S0, K, T, sigma, r):
    X_new = np.asarray([[K, T, S0, sigma, r]])
    y_pred = XGBr_model_eu_puts.predict(X_new)
    return y_pred[0]

# print(XGBr_eu_puts(230, 246/365, 189.71, 0.3318, 0.269))
'''
DTR
'''

def DTR_am_puts(S0, K, T, sigma, r):
    X_new = np.asarray([[K, T, S0, sigma, r]])
    y_pred = DTR_model_am_p.predict(X_new)
    return y_pred[0]

# print(DTR_am_puts(230, 246/365, 189.71, 0.3318, 0.269))

def DTR_am_calls(S0, K, T, sigma, r):
    X_new = np.asarray([[K, T, S0, sigma, r]])
    y_pred = DTR_model_am_c.predict(X_new)
    return y_pred[0]

# print(DTR_am_calls(230, 246/365, 189.71, 0.3318, 0.269))

def DTR_eu_puts(S0, K, T, sigma, r):
    X_new = np.asarray([[K, T, S0, sigma, r]])
    y_pred = DTR_model_eu_p.predict(X_new)
    return y_pred[0]

# print(DTR_am_puts(230, 246/365, 189.71, 0.3318, 0.269))

def DTR_eu_calls(S0, K, T, sigma, r):
    X_new = np.asarray([[K, T, S0, sigma, r]])
    y_pred = DTR_model_eu_c.predict(X_new)
    return y_pred[0]

# print(DTR_am_calls(230, 246/365, 189.71, 0.3318, 0.269))