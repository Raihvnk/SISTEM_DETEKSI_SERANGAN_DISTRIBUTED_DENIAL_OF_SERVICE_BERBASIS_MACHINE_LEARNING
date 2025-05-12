# %%
import pandas as pd

# Import dataset
dataset = pd.read_csv('preprocessed_train_datasets.csv')

# %%
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Memisahkan fitur dataset (X) dan class/label dataset (y)
X = dataset.drop('label', axis=1)
y = dataset['label']

# random_state=1 artinya tanpa random
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create and train the SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Menghitung dan mencetak akurasi model dengan metode Holdout
modelScore = svm_model.score(X_test, y_test)
print("Holdout score: ", modelScore)


# %%
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Memisahkan fitur dataset (X) dan class/label dataset (y)
X = dataset.drop('label', axis=1) # atau menggunakan X = dataset.drop('class', axis=1)
y = dataset['label'] # atau menggunakan y = dataset['class'

# random_state=42 adalah untuk reproduksibilitas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Menset model ke model yang dipilih dan mem‐fit modelnya
# model dapat diganti secara manual
model = SVC(kernel='linear') # menggunakan Support Vector Machine dengan kernel linear

# Melatih (fit) model menggunakan X_train, y_train data
model.fit(X_train, y_train)

# Menghitung dan mencetak akurasi model dengan metode Random Subsampling (random_state=42)
modelScore = model.score(X_test, y_test)
print("Random Subsampling score: ", modelScore)


# %%
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import numpy as np

# Define the mapping from labels to numeric values
label_mapping = {
    'NORMAL_TCP': 1,
    'DDOS_TCP': 2,
    'DDOS_UDP': 3,
    'NORMAL_UDP': 4,
    'NORMAL_ICMP': 5,
    'DDOS_ICMP': 6
}

# Apply the mapping to the 'label' column
dataset['label'] = dataset['label'].map(label_mapping)

# Memisahkan fitur dataset (X) dan class/label dataset (y)
X = dataset.drop('label', axis=1) # atau menggunakan X = dataset.drop('class', axis=1)
y = dataset['label'] # atau menggunakan y = dataset['class'

# Menset model ke model yang dipilih dan mem‐fit modelnya
# model dapat diganti secara manual
model = SVC(kernel='linear') # menggunakan Support Vector Machine dengan kernel linear

# Menghitung dan mencetak akurasi model
kFoldValidation = KFold(10)
modelScore = cross_val_score(model, X, y, cv=kFoldValidation)
print("K‐Fold Cross Validation score: ", modelScore) # hasilnya score sebanyak K
print("Rata-rata KFCV score: ", np.mean(modelScore)) # menghitung rata-rata score dari score sejumlah K

"""
Kadang kala diperlukan ukuran error dari model sebagai kebalikan dari ukuran akurasi.
Berikut ini adalah menghitung error menggunakan MEA dan RMSE
"""
# Menghitung dan mencetak Mean Absolute Error (MAE) model
maeScore = cross_val_score(model, X, y, cv=kFoldValidation, scoring='neg_mean_absolute_error')
print("K‐Fold Cross Validation MAE: ", maeScore) # hasilnya mae sebanyak K
print("Rata-rata KFCV Mean Absolute Error: ", np.mean(maeScore)) # menghitung rata-rata mae dari mae sejumlah K

# Menghitung dan mencetak Root Mean Square Error (RMSE) model
rmseScore = cross_val_score(model, X, y, cv=kFoldValidation, scoring='neg_mean_squared_error')
print("K‐Fold Cross Validation RMSE: ", rmseScore) # hasilnya rmse sebanyak K
print("Rata-rata KFCV Root Mean Square Error: ", np.mean(rmseScore)) # menghitung rata-rata rmse dari mae sejumlah K



