from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2

def arch1(inputs):
    name = 'arch1'
    output = Dense(10, activation='relu', name='FHL2')(inputs)
    return output

def arch2(inputs):
    name = 'arch2'
    layer1 = Dense(24, activation='relu', name='FHL1')(inputs)
    layer2 = Dense(16, activation='relu', name='FHL2')(layer1)
    output = Dense(8, activation='relu', name='FHL3')(layer2)
    return output
