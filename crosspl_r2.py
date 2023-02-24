import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

df = pd.read_csv('preds.csv')
print(df.head(50))

anchor_pred = df['Y_predm_1'].values
anchor_true= df['Y_true_1'].values
biof_pred = df['Y_predm_0'].values
biof_true= df['Y_true_0'].values

r2_anchor = r2_score(anchor_true, anchor_pred)
r2_biof = r2_score(biof_true, biof_pred)
print(r2_anchor, r2_biof)
