# %%
import pandas as pd
# membaca dataset 
dataset = pd.read_csv('preprocessed_datasets.csv', sep=',')

# %%
#Metode Holdout

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# Memisahkan fitur dataset (X) dan class/label dataset (y)
X = dataset.drop('label', axis=1) # or using X = dataset.drop('class', axis=1)
y = dataset['label'] # or using y = dataset['class']
# random_state=1 artinya tanpa random
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# Menset model ke model yang dipilih dan mem‐fit modelnya
# model dapat diganti secara manual
model = KNeighborsClassifier(n_neighbors=3) # algortima klasifikasi Nearest Neighbor
# Melatih (fit) model menggunakan X_train, y_train data
model.fit(X_train, y_train)
# Menghitung dan mencetak akurasi model dengan metode Holdout
modelScore = model.score(X_test, y_test)
print("Holdout score: ", modelScore)

# %%
#Random Subsampling

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# Memisahkan fitur dataset (X) dan class/label dataset (y)
X = dataset.drop('label', axis=1) # or using X = dataset.drop('class', axis=1)
y = dataset['label'] # or using y = dataset['class'
# random_state=42 is for reproducility artinya random level 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Menset model ke model yang dipilih dan mem‐fit modelnya
# model dapat diganti secara manual
model = KNeighborsClassifier(n_neighbors=3) # algortima klasifikasi Nearest Neighbor
# Melatih (fit) model menggunakan X_train, y_train data
model.fit(X_train, y_train)
# Menghitung dan mencetak akurasi model dengan metode Random Subsampling (random_state=42)
modelScore = model.score(X_test, y_test)
print("Random Subsampling score: ", modelScore)

# %%
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd

# Memisahkan fitur dataset (X) dan class/label dataset (y)
X = dataset.drop('label', axis=1)
y = dataset['label']

# Menset model ke model yang dipilih dan mem‐fit modelnya
model = KNeighborsClassifier(n_neighbors=3)

# Menghitung dan mencetak akurasi model
kFoldValidation = KFold(10)

# Using accuracy as the scoring metric
accuracy_scores = cross_val_score(model, X, y, cv=kFoldValidation, scoring='accuracy')
print("K‐Fold Cross Validation Accuracy scores: ", accuracy_scores)
print("Ratas KFCV Accuracy score: ", np.mean(accuracy_scores))

# Using precision as the scoring metric
precision_scores = cross_val_score(model, X, y, cv=kFoldValidation, scoring='precision_macro')
print("K‐Fold Cross Validation Precision scores: ", precision_scores)
print("Ratas KFCV Precision score: ", np.mean(precision_scores))

# Using recall as the scoring metric
recall_scores = cross_val_score(model, X, y, cv=kFoldValidation, scoring='recall_macro')
print("K‐Fold Cross Validation Recall scores: ", recall_scores)
print("Ratas KFCV Recall score: ", np.mean(recall_scores))

# Using f1 as the scoring metric
f1_scores = cross_val_score(model, X, y, cv=kFoldValidation, scoring='f1_macro')
print("K‐Fold Cross Validation F1 scores: ", f1_scores)
print("Ratas KFCV F1 score: ", np.mean(f1_scores))



