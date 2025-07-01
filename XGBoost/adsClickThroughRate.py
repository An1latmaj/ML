import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score

# Load data
n_rows = 300000
df = pd.read_csv("train", nrows=n_rows)

# Prepare features and target
X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values

# Split into train and test sets
n_train = int(n_rows * 0.9)
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]

# One-hot encode categorical features
enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train)
X_test_enc = enc.transform(X_test)

# Train XGBoost model
model = xgb.XGBClassifier(learning_rate=0.1, max_depth=10, n_estimators=1000)
model.fit(X_train_enc, Y_train)

# Make predictions
pos_prob = model.predict_proba(X_test_enc)[:, 1]

# Evaluate model performance
print(f'The ROC AUC on testing set is: {roc_auc_score(Y_test, pos_prob):.3f}')
