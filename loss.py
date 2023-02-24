import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('metrics.csv')
epochs= np.arange(100)

plt.style.use('science')
plt.figure(figsize=(4,3))
plt.plot(epochs, df['loss'], 'k', label='Training')
plt.plot(epochs, df['val_loss'], 'r', label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(-4, 6)
plt.legend(loc='best')
plt.savefig('losses.pdf')
