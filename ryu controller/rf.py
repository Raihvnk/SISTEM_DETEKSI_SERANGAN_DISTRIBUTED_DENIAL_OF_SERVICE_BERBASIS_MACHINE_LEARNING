import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV
import joblib
from scipy.stats import randint

# Load dataset
train_dataset = pd.read_csv('preprocessed_train_datasets.csv')
test_dataset = pd.read_csv('preprocessed_test_datasets.csv')

# Separate features and labels
X_train = train_dataset.drop('label', axis=1)
y_train = train_dataset['label']
X_test = test_dataset.drop('label', axis=1)
y_test = test_dataset['label']

# Save feature names for later use
feature_names = X_train.columns.tolist()
joblib.dump(feature_names, 'model_features.pkl')

# Hyperparameter tuning
param_dist = {
	'n_estimators': randint(50, 200),
	'max_depth': [None, 10, 20, 30],
	'min_samples_split': randint(2, 11),
	'min_samples_leaf': randint(1, 5),
	'bootstrap': [True, False]
}

# Initialize model
rf = RandomForestClassifier(random_state=42)

# Randomized search
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=50, cv=3, n_jobs=-1, verbose=2, random_state=42)
random_search.fit(X_train, y_train)

# Best model
best_rf = random_search.best_estimator_

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracy_scores = []
confusion_matrices = []
classification_reports = []
roc_auc_scores = []

for train_index, val_index in kf.split(X_train):
	X_train_kfold, X_val_kfold = X_train.iloc[train_index], X_train.iloc[val_index]
	y_train_kfold, y_val_kfold = y_train.iloc[train_index], y_train.iloc[val_index]
    
	best_rf.fit(X_train_kfold, y_train_kfold)
    
	accuracy_scores.append(best_rf.score(X_val_kfold, y_val_kfold))
    
	y_val_pred = best_rf.predict(X_val_kfold)
	y_val_pred_proba = best_rf.predict_proba(X_val_kfold)[:, 1]
    
	confusion_matrices.append(confusion_matrix(y_val_kfold, y_val_pred))
	classification_reports.append(classification_report(y_val_kfold, y_val_pred))
	roc_auc_scores.append(roc_auc_score(y_val_kfold, best_rf.predict_proba(X_val_kfold), multi_class='ovr'))

mean_accuracy = np.mean(accuracy_scores)
mean_conf_matrix = np.mean(confusion_matrices, axis=0)
mean_roc_auc = np.mean(roc_auc_scores)

print("K-Fold Cross-Validation Mean Accuracy: ", mean_accuracy)
print("K-Fold Cross-Validation Mean Confusion Matrix:")
print(mean_conf_matrix)
print("K-Fold Cross-Validation Mean ROC AUC Score:")
print(mean_roc_auc)

for i, class_report in enumerate(classification_reports):
	print(f"Classification Report Fold {i+1}:")
	print(class_report)

print("\nFinal Evaluation on Test Dataset:")

y_test_pred = best_rf.predict(X_test)
y_test_pred_proba = best_rf.predict_proba(X_test)[:, 1]

test_conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Test Confusion Matrix:")
print(test_conf_matrix)

test_class_report = classification_report(y_test, y_test_pred)
print("Test Classification Report:")
print(test_class_report)

test_roc_auc = roc_auc_score(y_test, best_rf.predict_proba(X_test), multi_class='ovr')
print("Test ROC AUC Score:")
print(test_roc_auc)

# Feature Importance
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for f in range(X_train.shape[1]):
	print(f"{f + 1}. feature {indices[f]} ({importances[indices[f]]})")

# Save the model
joblib.dump(best_rf, 'model.pkl')




