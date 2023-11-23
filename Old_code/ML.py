from keras.models import load_model
import numpy as np

# Load the saved model
model_american_calls = load_model("model_american_calls.h5")

# Use the model to predict on new data samples
X_new = np.asarray([[230, 246/365, 189.71, 0.3318, 0.269]])
y_pred = model_american_calls.predict(X_new)
print(y_pred)