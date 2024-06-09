import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
# readingdata
filePath = 'processed_dataset.csv'
dfCleaning = pd.read_csv(filePath)
# target and value 
X = dfCleaning.drop(columns=['aggregateRating/ratingValue'])
y = dfCleaning['aggregateRating/ratingValue']
# standscaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
# data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# model training
rf_model = RandomForestClassifier(random_state=10, oob_score=True)
rf_model.fit(X_train, y_train)
# pred
y_pred = rf_model.predict(X_test)
y_pred_prob = rf_model.predict_proba(X_test)
#counting value
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix')
print(cm)
print('----------------')
accuracy = accuracy_score(y_test, y_pred)
print('Test Accuracy:', accuracy)
print('----------------')

try:
    auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
    print('ROC AUC Score:', auc)
except ValueError as e:
    print(f'Error calculating ROC AUC Score: {e}')
print('----------------')

ps = precision_score(y_test, y_pred, average='weighted', zero_division=1)
rs = recall_score(y_test, y_pred, average='weighted')
f1sc = f1_score(y_test, y_pred, average='weighted')
print('Precision Score:', ps)
print('Recall Score:', rs)
print('F1 Score:', f1sc)
#draw auc 
plt.figure(figsize=(10, 7))
for i, color in zip(range(len(rf_model.classes_)), ['blue' , 'orange', 'green']):
    fpr, tpr, __  = roc_curve(y_test, y_pred_prob[: , i], pos_label=rf_model.classes_[i])
    plt.plot(fpr[i], tpr[i], color = color, lw = 2, label = )
