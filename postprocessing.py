import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tfk = tf.keras
tf.keras.backend.set_floatx("float64")
import tensorflow_probability as tfp
tfd = tfp.distributions
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

def plot_losses(model, inputs, outputs, data_test, arch, co=''):
    metrics = pd.DataFrame(model.history.history)
    #metrics['loss-log']=np.log10(metrics['loss'])
    #metrics['val-loss-log']=np.log10(metrics['val_loss'])
        
    metrics.to_csv('metrics'+str(arch.__name__)+'.csv', index=False)

    plt.style.use('science')
    plt.figure(figsize=(4,3))
    plt.plot(np.arange(len(metrics)), metrics['loss'], 'k', label='Training')
    plt.plot(np.arange(len(metrics)), metrics['val_loss'], 'r', label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Log($L$)')
    plt.legend(loc='best')

    plt.savefig('losses'+str(arch.__name__)+'.pdf')

    further = 1
    if further:
        # Predict.
        samples = 1000
        iterations = 10
        test_iterator = tf.compat.v1.data.make_one_shot_iterator(data_test)
        X_true, Y_true, Y_pred = np.empty(shape=(samples, len(inputs))), np.empty(shape=(samples, len(outputs))), np.empty(shape=(samples, len(outputs), iterations))

        for i in range(samples):
            features, labels = test_iterator.get_next()
            X_true[i,:] = features
            Y_true[i,:] = labels.numpy()
            for k in range(iterations):
                Y_pred[i,:,k] = model.predict(features)

        # Calculate mean and standard deviation.
        Y_pred_m = np.mean(Y_pred, axis=-1)
        Y_pred_s = np.std(Y_pred, axis=-1)

        dg = pd.DataFrame()
        dg['Y_true_0'] = Y_true[:,0]
        dg['Y_predm_0'] = Y_pred_m[:,0]
        dg['Y_preds_0'] = Y_pred_s[:,0]
        dg['Y_true_1'] = Y_true[:,1]
        dg['Y_predm_1'] = Y_pred_m[:,1]
        dg['Y_preds_1'] = Y_pred_s[:,1]

        dg.to_csv('preds.csv', index=False)

        anchor_pred = dg['Y_predm_1'].values
        anchor_true= dg['Y_true_1'].values
        biof_pred = dg['Y_predm_0'].values
        biof_true= dg['Y_true_0'].values

        r2_anchor = r2_score(anchor_true, anchor_pred)
        r2_biof = r2_score(biof_true, biof_pred)

        print(r2_anchor, r2_biof)

        with open('network_scores'+str(co)+'.txt', 'a') as f:
            f.write('{}, {:.5f}, {:.5f}\n'.format(arch.__name__, r2_biof, r2_anchor))
