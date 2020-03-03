# -*- coding: utf-8 -*-

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("lenses.csv")

x_train, x_test, y_train, y_test = train_test_ split(data.loc[::,['Edad', 'Lentes', 'Astigmatismo', 'Lagrimeo']], data['Tipo'])

clf = SVC()

clf.fit(x_train, y_train)

print(clf.score(x_test, y_test))
