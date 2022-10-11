#APARTADO C

from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
#%matplotlib notebook
from matplotlib import pyplot as plt
import scipy.stats

# Visualitzarem només 3 decimals per mostra
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Funcio per a llegir dades en format csv
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

# Carreguem dataset d'exemple
dataset = load_dataset('dummy.csv')
data = dataset.values

x = data[:, 0:34]
y = data[:, 33]
#x = data[:, :2]
#y = data[:, 2]

print("Dimensionalitat de la BBDD:", dataset.shape)
print("Dimensionalitat de les entrades X", x.shape)
print("Dimensionalitat de l'atribut Y", y.shape)
print("###########################################################")
print("Per comptar el nombre de valors no existents:")
print(dataset.isnull().sum())
print("###########################################################")
print("Per visualitzar les primeres 5 mostres de la BBDD:")
print(dataset.head())
print("###########################################################")
print("Per veure estadístiques dels atributs numèrics de la BBDD:")
print(dataset.describe())

# mostrem atribut 0
plt.figure()
ax = plt.scatter(x[:,3], y)
plt.axhline(y=-0.5, xmin=0.0, xmax=1.0)

xaux = np.linspace(-1, 3, 10)
yaux = - xaux -0.5
plt.plot(xaux, yaux, "r-")
#plt.show()

plt.figure()
plt.title("Histograma de l'atribut 9")
plt.xlabel("Attribute Value")
plt.ylabel("Count")
hist = plt.hist(x[:,9], bins=11, range=[np.min(x[:,9]), np.max(x[:,9])], histtype="bar", rwidth=0.8)
plt.show()