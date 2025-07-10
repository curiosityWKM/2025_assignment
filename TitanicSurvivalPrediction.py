# -*- coding: utf-8 -*-
# %% [markdown]

"""
Homework:

The folder '~//data//homework' contains data of Titanic with various features and survivals.

Try to use what you have learnt today to predict whether the passenger shall survive or not.

Evaluate your model.
"""
# %%
# load data
import pandas as pd

data = pd.read_csv('data//train.csv')
df = data.copy()
df.sample(10)
# %%
# delete some features that are not useful for prediction
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
df.info()
# %%
# check if there is any NaN in the dataset
print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))
df.dropna(inplace=True)
print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))
# %%
# convert categorical data into numerical data using one-hot encoding
# For example, a feature like sex with categories ['male', 'female'] would be transformed into two new binary features, sex_male and sex_female, represented by 0 and 1.
df = pd.get_dummies(df)
df.sample(10)
# %% 
# separate the features and labels
X = df.drop(columns=['Survived'])
y = df['Survived']

# %%
# train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

# %%
# build model
# build three classification models
# SVM, KNN, Random Forest
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
svm_model = SVC(kernel='linear', random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# fit the models
svm_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# %%
# predict and evaluate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# SVM
y_pred_svm = svm_model.predict(X_test)
print('SVM Accuracy: {:.2f}%'.format(accuracy_score(y_test, y_pred_svm) * 100))
print('SVM Classification Report:\n', classification_report(y_test, y_pred_svm))
print('SVM Confusion Matrix:\n', confusion_matrix(y_test, y_pred_svm))
# KNN
y_pred_knn = knn_model.predict(X_test)
print('KNN Accuracy: {:.2f}%'.format(accuracy_score(y_test, y_pred_knn) * 100))
print('KNN Classification Report:\n', classification_report(y_test, y_pred_knn))
print('KNN Confusion Matrix:\n', confusion_matrix(y_test, y_pred_knn))
# Random Forest
y_pred_rf = rf_model.predict(X_test)
print('Random Forest Accuracy: {:.2f}%'.format(accuracy_score(y_test, y_pred_rf) * 100))
print('Random Forest Classification Report:\n', classification_report(y_test, y_pred_rf))
print('Random Forest Confusion Matrix:\n', confusion_matrix(y_test, y_pred_rf))

