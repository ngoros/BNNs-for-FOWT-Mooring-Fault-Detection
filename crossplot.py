import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('preds.csv')

df['Y_true_0'] = df['Y_true_0']+abs(np.min(df['Y_true_0']))
df['Y_true_1'] = df['Y_true_1']+abs(np.min(df['Y_true_1']))
df['Y_predm_0'] = df['Y_predm_0']+abs(np.min(df['Y_predm_0']))
df['Y_predm_1'] = df['Y_predm_1']+abs(np.min(df['Y_predm_1']))
#df['Y_preds_0'] = df['Y_preds_0']+abs(np.min(df['Y_preds_0']))
#df['Y_preds_1'] = df['Y_preds_1']+abs(np.min(df['Y_preds_1']))

damage = 'biofouling'
if damage == 'biofouling':
    df = df.sort_values('Y_true_0', ascending=True)
    
else:
    df = df.sort_values('Y_true_1', ascending=True)

y0_true = df['Y_true_0']/np.max(df['Y_true_0'])
y1_true = df['Y_true_1']/np.max(df['Y_true_1'])
y0_pred = df['Y_predm_0']/np.max(df['Y_predm_0'])
y1_pred = df['Y_predm_1']/np.max(df['Y_predm_1'])
y0_s = df['Y_preds_0']#/np.max(df['Y_preds_0'])
y1_s = df['Y_preds_1']#/np.max(df['Y_preds_1'])

y0_upper = y0_pred + 2*y0_s
y0_lower = y0_pred - 2*y0_s
y1_upper = y1_pred + 2*y1_s
y1_lower = y1_pred - 2*y1_s

if damage == 'biofouling':
    plt.style.use('science')
    plt.figure(figsize=(4,4))
    plt.plot(y0_true, y0_pred,'k', linewidth=1, marker='.', markersize=0)
    plt.plot(y0_true, y0_upper,'k--', linewidth=0.2, marker='.', markersize=0)
    plt.plot(y0_true, y0_lower,'k--', linewidth=0.2, marker='.', markersize=0)
    plt.fill_between(y0_true, y0_upper, y0_lower, color='yellow', alpha=0.3)
    plt.plot(np.linspace(0,1,5), np.linspace(0,1,5), 'r--')
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig('biof_crossp.pdf')

else:
    plt.style.use('science')
    plt.figure(figsize=(4,4))
    plt.plot(y1_true, y1_pred,'k', linewidth=1, marker='.', markersize=0)
    plt.plot(y1_true, y1_upper,'k--', linewidth=0.2, marker='.', markersize=0)
    plt.plot(y1_true, y1_lower,'k--', linewidth=0.2, marker='.', markersize=0)
    plt.fill_between(y1_true, y1_upper, y1_lower, color='yellow', alpha=0.3)
    plt.plot(np.linspace(0,1,5), np.linspace(0,1,5), 'b--')
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig('anchor_crossp.pdf')
