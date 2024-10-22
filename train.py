import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier

train_file = 'train.parquet'
model_file = 'model.joblib'

train_df = pd.read_parquet(train_file)
train_df['label'] = train_df['label'].astype('float32')

top_k = 15

def extract_features(values):
    fft_vals = np.fft.fft(values)
    top_freqs = np.argsort(np.abs(fft_vals))[-top_k:]
    return np.log(np.abs(fft_vals[top_freqs]))

X_train = np.nan_to_num(np.array([extract_features(row) for row in train_df['values']]), nan=0)
y_train = train_df['label'].values

clf = GradientBoostingClassifier(
    subsample=0.6,
    n_estimators=100,
    min_samples_leaf=36,
    max_leaf_nodes=50,
    max_depth=10,
    learning_rate=0.01,
)

clf.fit(X_train, y_train)

joblib.dump(clf, model_file)
