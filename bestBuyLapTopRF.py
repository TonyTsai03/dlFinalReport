import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

filePath = 'processed_dataset.csv'
dfCleaning =  pd.read_csv(filePath)

#target and value
X = dfCleaning.drop(columns=['aggregateRating/ratingValue'])
y = dfCleaning['aggregateRating/ratingValue']

#standScaler or labelencoder

#SMOTE

#data split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1)

#feeding data
random = 40
nEstimators = 100
rfmodel = RandomForestClassifier(n_estimators = nEstimators, random_state = random)
rfmodel.fit(X_train, y_train)

#testing 
yPred = rfmodel.predict(X_test)
yPredProba = rfmodel.predict_proba(X_test)

#data and form
accuracyValue = accuracy_score(X_test, yPred)