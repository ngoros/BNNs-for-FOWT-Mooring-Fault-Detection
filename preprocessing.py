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
# Define helper functions
scaler = MinMaxScaler()
detector = IsolationForest(n_estimators=1000, contamination="auto", random_state=0)
neg_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)

def read_data(path, n_epochs=100, n_batches=10):
    # Load data
    data = pd.read_csv("data/joint.csv", header=None)

    # Select columns and remove rows with missing values.
    columns = [
            'Hs', 'Tp', 'V',
            'Mean-Surge', 'SD-Surge', 'F1-Surge', 'F2-Surge', 'M-Surge',
            'Mean-Sway', 'SD-Sway', 'F1-Sway', 'F2-Sway', 'M-Sway',
            'Mean-Heave', 'SD-Heave', 'F1-Heave', 'F2-Heave', 'M-Heave',
            'Mean-Roll', 'SD-Roll', 'F1-Roll', 'F2-Roll', 'M-Roll',
            'Mean-Pitch', 'SD-Pitch', 'F1-Pitch', 'F2-Pitch', 'M-Pitch',
            'Mean-Yaw', 'SD-Yaw', 'F1-Yaw', 'F2-Yaw', 'M-Yaw',
            'C1', 'C2']

    data.columns = columns
    data = data[columns].dropna(axis=0)

    # Scale data to zero mean and unit variance.
    X_t = scaler.fit_transform(data)
    dataset = pd.DataFrame(X_t, columns=columns)

    # Select labels for inputs and outputs.
    inputs = ['Hs', 'Tp', 'V',
            'Mean-Surge', 'SD-Surge', 'F1-Surge', 'F2-Surge', 'M-Surge',
            'Mean-Sway', 'SD-Sway', 'F1-Sway', 'F2-Sway', 'M-Sway',
            'Mean-Heave', 'SD-Heave', 'F1-Heave', 'F2-Heave', 'M-Heave',
            'Mean-Roll', 'SD-Roll', 'F1-Roll', 'F2-Roll', 'M-Roll',
            'Mean-Pitch', 'SD-Pitch', 'F1-Pitch', 'F2-Pitch', 'M-Pitch',
            'Mean-Yaw', 'SD-Yaw', 'F1-Yaw', 'F2-Yaw', 'M-Yaw']

    outputs = ['C1', 'C2']

    # Define some hyperparameters.
    n_samples = dataset.shape[0]
    batch_size = np.floor(n_samples/n_batches)
    buffer_size = n_samples                                                                             # Define training and test data sizes.
    n_train = int(0.8*dataset.shape[0])                                                                 # Define dataset instance.
    data = tf.data.Dataset.from_tensor_slices((dataset[inputs].values, dataset[outputs].values))
    data = data.shuffle(n_samples, reshuffle_each_iteration=True)                                       # Define train and test data instances.
    data_train = data.take(n_train).batch(batch_size).repeat(n_epochs)
    data_test = data.skip(n_train).batch(1)

    return data, data_train, data_test, inputs, outputs
