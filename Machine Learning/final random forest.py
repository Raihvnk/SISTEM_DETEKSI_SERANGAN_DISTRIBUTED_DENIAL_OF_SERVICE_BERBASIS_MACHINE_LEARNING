# %%
# %%
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
from sklearn.model_selection import KFold
# Memuat dataset yang sudah di-split
train_dataset = pd.read_csv('preprocessed_train_datasets.csv')
test_dataset = pd.read_csv('preprocessed_test_datasets.csv')
# %%
# Memisahkan fitur dan label untuk data training dan testing
X_train = train_dataset.drop('label', axis=1)
y_train = train_dataset['label']
X_test = test_dataset.drop('label', axis=1)
y_test = test_dataset['label']
# Menyimpan nama fitur untuk penggunaan nanti di Ryu controller
feature_names = X_train.columns.tolist()
# Metode K-Fold Cross-Validation
n_splits = 5 # Tentukan jumlah lipatan (folds)
# Inisialisasi K-Fold
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
# Inisialisasi variabel untuk menyimpan hasil evaluasi
accuracy_scores = []
confusion_matrices = []
classification_reports = []
roc_auc_scores = []
# Loop melalui setiap fold
for train_index, val_index in kf.split(X_train):
 # Membagi data ke dalam data training dan data validasi
 X_train_kfold, X_val_kfold = X_train.iloc[train_index], X_train.iloc[val_index]
 y_train_kfold, y_val_kfold = y_train.iloc[train_index], y_train.iloc[val_index]

 # Membuat dan melatih model Random Forest
 rf_model_kfold = RandomForestClassifier() # Anda dapat menyesuaikan
#parameter model sesuai kebutuhan
 rf_model_kfold.fit(X_train_kfold, y_train_kfold)

 # Evaluasi model pada data validasi
 accuracy_scores.append(rf_model_kfold.score(X_val_kfold, y_val_kfold))

 # Membuat prediksi pada data validasi
 y_val_pred = rf_model_kfold.predict(X_val_kfold)
 y_val_pred_proba = rf_model_kfold.predict_proba(X_val_kfold)[:, 1]

 # Menghitung dan menyimpan confusion matrix
 conf_matrix_kfold = confusion_matrix(y_val_kfold, y_val_pred)
 confusion_matrices.append(conf_matrix_kfold)

 # Menghitung dan menyimpan classification report
 class_report_kfold = classification_report(y_val_kfold, y_val_pred)
 classification_reports.append(class_report_kfold)

 # Menghitung dan menyimpan ROC AUC score
 roc_auc_kfold = roc_auc_score(y_val_kfold,
rf_model_kfold.predict_proba(X_val_kfold), multi_class='ovr')
 roc_auc_scores.append(roc_auc_kfold)
# Menghitung rata-rata hasil evaluasi
mean_accuracy = np.mean(accuracy_scores)
mean_conf_matrix = np.mean(confusion_matrices, axis=0)
mean_roc_auc = np.mean(roc_auc_scores)
# Mencetak hasil evaluasi
print("K-Fold Cross-Validation Mean Accuracy: ", mean_accuracy)
print("K-Fold Cross-Validation Mean Confusion Matrix:")
print(mean_conf_matrix)
print("K-Fold Cross-Validation Mean ROC AUC Score:")
print(mean_roc_auc)
# Jika ingin mencetak classification report dari setiap fold
for i, class_report in enumerate(classification_reports):
 print(f"Classification Report Fold {i+1}:")
 print(class_report)
# Evaluasi akhir menggunakan test_dataset
print("\nFinal Evaluation on Test Dataset:")
# Membuat prediksi menggunakan test_dataset
y_test_pred = rf_model_kfold.predict(X_test)
y_test_pred_proba = rf_model_kfold.predict_proba(X_test)[:, 1]
# Menghitung dan mencetak confusion matrix untuk test data
test_conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Test Confusion Matrix:")
print(test_conf_matrix)
# Menghitung dan mencetak classification report (precision, recall, F1-score) untuk test data
test_class_report = classification_report(y_test, y_test_pred)
print("Test Classification Report:")
print(test_class_report)
# Menghitung dan mencetak ROC AUC score untuk test data
test_roc_auc = roc_auc_score(y_test, rf_model_kfold.predict_proba(X_test),
multi_class='ovr')
print("Test ROC AUC Score:")
print(test_roc_auc)


