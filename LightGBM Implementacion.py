# LightGBM

# Installing LightGBM
# Open a terminal (Mac & Linux) or the anaconda prompt (Windows) and enter the following command:
# conda install -c conda-forge lightgbm

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#onehotencoder = OneHotEncoder(categorical_features = [1])
#X = OneHotEncoder.fit_transform(X).toarray()
#X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting LightGBM to the Training set
import lightgbm as lgb
training_data = lgb.Dataset(data = X_train, label = y_train)
params = {'num_leaves': 31, 'num_trees': 100, 'objective': 'binary'}
params['metric'] = ['auc', 'binary_logloss']
classifier = lgb.train(params = params,
                       train_set = training_data,
                       num_boost_round = 100)

# Predicting the Test set results
prob_pred = classifier.predict(X_test)
y_pred = np.zeros(len(prob_pred))
for i in range(0, len(prob_pred)):
    if prob_pred[i] >= 0.5:
       y_pred[i] = 1
    else:  
       y_pred[i] = 0

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Getting the Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_pred, y_test) * 100
print("Accuracy: {:.0f} %".format(accuracy))

# Applying k-Fold Cross Validation
params = {'num_leaves': 31, 'num_trees': 100, 'objective': 'binary'}
params['metric'] = ['auc']
cv_results = lgb.cv(params = params,
                    train_set = training_data,
                    num_boost_round = 10,
                    nfold = 10)
average_auc = np.mean(cv_results['auc-mean'])
print("Average AUC: {:.0f} %".format(accuracy))