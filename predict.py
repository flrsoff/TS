import pandas as pd
import numpy as np
import joblib

test_file = 'test.parquet'
result_file = 'submission.csv'
model_file = 'model.joblib'

test_df = pd.read_parquet(test_file)

top_k = 15

def extract_features(values):
    fft_vals = np.fft.fft(values)
    top_freqs = np.argsort(np.abs(fft_vals))[-top_k:]
    return np.log(np.abs(fft_vals[top_freqs]))

X_test = np.nan_to_num(np.array([extract_features(row) for row in test_df['values']]), nan=0)

clf = joblib.load(model_file)

y_pred = clf.predict_proba(X_test)[:, 1]

pred_df = pd.DataFrame({
    'id': test_df['id'],
    'score': y_pred
})
pred_df.to_csv(result_file, index=False)

