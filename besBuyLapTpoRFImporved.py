import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Load data
filePath = 'processed_dataset.csv'
dfCleaning = pd.read_csv(filePath)

# Split features and target variable
X = dfCleaning.drop(columns=['aggregateRating/ratingValue'])
y = dfCleaning['aggregateRating/ratingValue']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

# Define the model
rf_model = RandomForestClassifier(random_state=42, oob_score=True)

# Define the pipeline

# Define parameter grid for GridSearchCV
param_grid = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [None, 10, 20, 30],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4]
}

# Define k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=kf, n_jobs=-1, scoring='accuracy')

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters and estimator
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)
y_pred_prob = best_model.predict_proba(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix')
print(cm)
print('----------------')

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Test Accuracy:', accuracy)
print('----------------')

# ROC AUC Score
try:
    auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
    print('ROC AUC Score:', auc)
except ValueError as e:
    print(f'Error calculating ROC AUC Score: {e}')
print('----------------')

# Other metrics
ps = precision_score(y_test, y_pred, average='weighted', zero_division=1)
rs = recall_score(y_test, y_pred, average='weighted')
f1sc = f1_score(y_test, y_pred, average='weighted')
print('Precision Score:', ps)
print('Recall Score:', rs)
print('F1 Score:', f1sc)

# Plot ROC Curve
plt.figure(figsize=(10, 7))
for i in range(len(best_model.named_steps['rf'].classes_)):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob[:, i], pos_label=best_model.named_steps['rf'].classes_[i])
    plt.plot(fpr, tpr, label=f'Class {best_model.named_steps['rf'].classes_[i]} (AUC={auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RF ROC Curve')
plt.legend(loc='best')
plt.show()

# Residual plot
y_test_pred = best_model.predict(X_test)
y_train_pred = best_model.predict(X_train)

plt.figure(figsize=(10, 7))
plt.scatter(y_train_pred, y_train_pred - y_train,
            c='blue', marker='o',
            label='Training data', s=100, alpha=1)
plt.scatter(y_test_pred, y_test_pred - y_test,
            c='lightgreen', marker='s',
            label='Test data', s=100, alpha=0.5)
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0.5, xmin=min(min(y_train_pred)-0.5, min(y_test_pred))-0.5, xmax=max(max(y_train_pred)+0.5, max(y_test_pred)+0.5), color='black', lw=2)
plt.xlim([min(min(y_train_pred)-0.5, min(y_test_pred))-0.5, max(max(y_train_pred)+0.5, max(y_test_pred)+0.5)])
plt.tight_layout()
plt.show()
