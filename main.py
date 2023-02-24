import os
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
from src.preprocessing import read_data
from src.model import run_training
from src.postprocessing import *
from src.arch import *

# DEBUG FLAG
debug = False

# PREDEFINED PARAMETERS
n_epochs = 100
n_batches = 10

co=''
with open('network_scores'+str(co)+'.txt', 'w+') as f:
    f.close()

with open('network_scores'+str(co)+'.txt', 'w') as f:
    f.write('arch_name, r2_biof, r2_anchor\n')

# READ DATA
data_path = os.path.join('data', 'joint.csv')
data, data_train, data_test, inputs, outputs = read_data(data_path, n_epochs, n_batches)

archs = [arch1]

if debug:
    print('\nDATA READ SUCCESSFULLY\n')

for arch in archs:
    model = run_training(inputs, outputs, n_batches, data, data_train, data_test, arch, n_epochs)

    if debug:
        print('\nBNN TRAINED SUCCESSFULLY\n')

    plot_losses(model, inputs, outputs, data_test, arch)

if debug:
    print('\nPROGRAM ENDED SUCCESSFULLY\n')
