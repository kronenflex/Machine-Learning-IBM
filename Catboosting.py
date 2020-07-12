# -*- coding: utf-8 -*-
"""
@author: DiegoIgnacioPavezOlave
"""
# importar el dataset
import pandas as pd
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#preprocesado de datos Encoding
from sklearn.preprocessing import LabelEncoder
labelencoder_x_1 = LabelEncoder()
X[:, 1] = labelencoder_x_1.fit_transform(X[:, 1])
labelencoder_x_2 = LabelEncoder()
X[:, 2] = labelencoder_x_2.fit_transform(X[:, 2]) 

#Separacion del entremiento test training con validacion cruzada
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# ajuste de catboos para entrenar el dataset
from catboost import CatBoostClassifier
classifier = CatBoostClassifier(iterations = 2,
                                depth = 2,
                                learning_rate = 1,
                                loss_function = 'Logloss',
                                logging_level = 'Verbose')
classifier.fit(X_train, y_train, cat_features = [1,2])

#prediciendo el test
y_pred = classifier.predict(X_test)
prob_pred = classifier.predict_proba(X_test)

#Matrix de confucion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (float((cm[0,0] + cm[1,1])/(cm[0,0] + cm[1,1] + cm[1,0] + cm[0,1])))*100
print("accuracy: {:.0f} %".format(accuracy))

#validacion cruzada
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Average Accuracy: {:.0f} %".format(accuracies.mean() * 100))
print("Standard Deviation: {:.2f} %".format(accuracies.std() * 100))
