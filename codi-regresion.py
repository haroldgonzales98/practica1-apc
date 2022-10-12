from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Visualitzarem només 3 decimals per mostra
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Funcio per a llegir dades en format csv
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

dataset = load_dataset('dummy.csv')
# Convertim els valors de l'atribut DOB de str a int
if (type(dataset["DOB"][0]) == str):
    for index, valor in enumerate(dataset["DOB"]):
        valor2 = datetime.strptime(valor, '%Y-%m-%d')
        new_value = int(valor2.strftime("%Y%m%d"))
        dataset["DOB"] = dataset["DOB"].replace(valor, new_value)

dataset['Gender'] = dataset['Gender'].replace(['m'], 0)
dataset['Gender'] = dataset['Gender'].replace(['f'], 1)

dataset['Degree'] = dataset['Degree'].replace("B.Tech/B.E.", 1)
dataset['Degree'] = dataset['Degree'].replace("MCA", 2)
dataset['Degree'] = dataset['Degree'].replace("M.Tech./M.E.", 3)
dataset['Degree'] = dataset['Degree'].replace("M.Sc. (Tech.)", 4)
# for index, valor in enumerate(dataset["Degree"]):
#     if (valor == 1 or valor == 2 or valor == 3 or valor == 4):
#         None
#     else:
#         dataset["Degree"] = dataset["Degree"].replace(valor, 0)

########dataset[] = dataset[].replace(, )

dataset["Specialization"] = dataset["Specialization"].replace("electronics and communication engineering", 1)
dataset["Specialization"] = dataset["Specialization"].replace("computer science & engineering", 2)
dataset["Specialization"] = dataset["Specialization"].replace("information technology", 3)
dataset["Specialization"] = dataset["Specialization"].replace("computer engineering", 4)
dataset["Specialization"] = dataset["Specialization"].replace("computer application", 5)
for index, valor in enumerate(dataset["Specialization"]):
    if (valor != 1 and valor != 2 and valor != 3 and valor != 4):
        dataset["Specialization"] = dataset["Specialization"].replace(valor, 0)

for index, valor in enumerate(dataset["CollegeState"]):
    if (valor == "Uttar Pradesh"):
        dataset["CollegeState"] = dataset["CollegeState"].replace(valor, 1)
    elif (valor == "Karnataka"):
        dataset["CollegeState"] = dataset["CollegeState"].replace(valor, 2)
    elif (valor == "Tamil Nadu"):
        dataset["CollegeState"] = dataset["CollegeState"].replace(valor, 3)
    elif (valor == "Telangana"):
        dataset["CollegeState"] = dataset["CollegeState"].replace(valor, 4)
    elif (valor == "Maharashtra"):
        dataset["CollegeState"] = dataset["CollegeState"].replace(valor, 5)
    else:
        dataset["CollegeState"] = dataset["CollegeState"].replace(valor, 0)

for index, valor in enumerate(dataset["GraduationYear"]):
    if (valor < 2000):
        dataset["GraduationYear"] = dataset["GraduationYear"].replace(valor, 2010)

#dataset.drop(['ID', 'Gender', 'DOB', '10board', '12graduation', '12board', 'Degree', 'Specialization', 'CollegeState',
#              'CollegeID', 'CollegeCityID', 'CollegeState', 'CollegeCityTier'], axis=1, inplace=True)
dataset.drop(['10board', '12board'], axis=1, inplace=True)
# dataset.replace(-1, np.NaN, inplace=True)
# print(dataset.isnull().sum())
# dataset.drop(['10board', '12board', 'Domain', 'ComputerProgramming', 'ElectronicsAndSemicon', 'ComputerScience',
#               'MechanicalEngg', 'ElectricalEngg', 'TelecomEngg', 'CivilEngg'], axis=1, inplace=True)
data = dataset.values

def mse(v1, v2):
    return ((v1 - v2)**2).mean()

def regression(x, y):
    # Creem un objecte de regressió de sklearn
    regr = LinearRegression()

    # Entrenem el model per a predir y a partir de x
    regr.fit(x, y)

    # Retornem el model entrenat
    return regr

def standarize(x_train):
    mean = x_train.mean(0)
    std = x_train.std(0)
    x_t = x_train - mean[None, :]
    x_t /= std[None, :]
    return x_t

size = data.shape[1] - 1
x = data[:,:size]
y = data[:, size]

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state = 1234, shuffle = True)

entreno = []
for i in range(x_train.shape[1]):
    entreno.append(regression(x_train[:,i].reshape(-1,1), y_train))
entreno = np.array(entreno)
predicciones = []
for i in range(x_test.shape[1]):
    predicciones.append(entreno[i].predict(x_test[:,i].reshape(-1,1)))

results = []
for i, pred in enumerate(predicciones):
    results.append("Atributo " + dataset.columns[i] + " : " + str(mse(standarize(pred.reshape(-1,1)), standarize(y_test.reshape(-1,1)))))

######################################################################################

data = standarize(data)
x_normal = data[:,:size]
y = data[:, size]

x_train, x_test, y_train, y_test = train_test_split(x_normal,y, train_size = 0.8, random_state = 1234, shuffle = True)

entreno_n = []
for i in range(x_train.shape[1]):
    entreno_n.append(regression(x_train[:,i].reshape(-1,1), y_train))
entreno_n = np.array(entreno_n)
predicciones_n = []
for i in range(x_test.shape[1]):
    predicciones_n.append(entreno_n[i].predict(x_test[:,i].reshape(-1,1)))

results_n = []
for i, pred in enumerate(predicciones_n):
    results_n.append("Atributo " + dataset.columns[i] + " : " + str(mse(pred, y_test)))

aux = dataset.columns[0]
print(x)