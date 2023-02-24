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
from src.arch import *

neg_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)


def run_training(inputs, outputs, n_batches, data, data_train, data_test, arch_fun, n_epochs=100):

    # Define prior for regularization.
    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(len(outputs), dtype=tf.float64), scale=1.0), reinterpreted_batch_ndims=1)

    # Define model instance.
    #model = tfk.Sequential([
    #    tfk.layers.InputLayer(input_shape=(len(inputs),), name="input"),
        #tfk.layers.Dense(10, activation="relu", name="dense_1"),
    #    inter_layers,
    #    tfk.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(len(outputs)), activation=None, name="distribution_weights"),
    #    tfp.layers.MultivariateNormalTriL(len(outputs), activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=1/n_batches), name="output")], name="model")

    l0 = tfk.Input(shape=(len(inputs),), name='input')
    output_H = arch_fun(l0)
    l_m = tfk.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(len(outputs)), activation=None, name="distribution_weights")(output_H)
    l_f = tfp.layers.MultivariateNormalTriL(len(outputs), activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=1/n_batches), name="output")(l_m)
    model = keras.Model(inputs=l0, outputs=l_f, name='MODEL')

    # Compile model.
    model.compile(optimizer="adam", loss=neg_log_likelihood)

    # Run training session.
    model.fit(data_train, epochs=n_epochs, validation_data=data_test, verbose=True)

    # Describe model.
    model.summary()

    return model

#    P = keras.Input(shape=(2,))
#    r = keras.Input(shape=(3,))
    #
    #layer1 = Dense(16, activation='relu', name='HL1')([P, r])
    #output_H = Dense(16, activation='relu', name='HL2')(layer1)
#    output_H, arch_name = arch_fun([P, r])
    #
#    u = Dense(30, activation='sigmoid')(output_H)
    # 
#    model = keras.Model(inputs=[P, r], outputs=u, name='FORWARD_MODEL')
