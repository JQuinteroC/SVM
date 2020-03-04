# -*- coding: utf-8 -*-

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

data = pd.read_csv("lenses.csv")

x_train, x_test, y_train, y_test = train_test_split(data.loc[::,['Edad', 'Lentes', 'Astigmatismo', 'Lagrimeo']], data['Tipo'])

clf = SVC(gamma = 'scale')

clf.fit(x_train, y_train)

print("precision media del modelo")
print(clf.score(x_test, y_test))

print("para una personaje joven, miope, astigmatica y con lagrimeo normal, tendra que usar unos lentes: ")
res = clf.predict(np.reshape(np.array([1,1,1,2]), (1,4)))
if res == 1:
    print("Lentes de contacto duras")
elif res == 2:
    print("Lentes de contacto blandas")
elif res == 3:
    print("No debe usar lentes de contacto")
else:
    print("No se puede determinar")