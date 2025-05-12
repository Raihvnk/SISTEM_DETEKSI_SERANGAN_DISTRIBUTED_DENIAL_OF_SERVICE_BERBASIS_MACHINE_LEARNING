# %%
# Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
# plt.style.use('default')
color_pallete = ['#fc5185', '#3fc1c9', '#364f6b']
sns.set_palette(color_pallete)
sns.set_style("white")
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# %%
import pandas as pd

# Import dataset
dataset = pd.read_csv('datasets.csv')

# Kelompokkan data berdasarkan label
grouped_data = dataset.groupby('label')

# Tampilkan 5 data pertama dari setiap label
for label, label_data in grouped_data:
  print(f"\nLabel: {label}")
  print(label_data.head().to_string(index=False))

# %%
#drop column
columns_to_drop = ['src_ip', 'dst_ip']
dataset = dataset.drop(columns=columns_to_drop)


# %%
#drop column
columns_to_drop = ['tos', 'flags', 'offset', 'code_icmp', 'rx_error_ave', 'rx_dropped_ave', 'tx_error_ave', 'tx_dropped_ave']
dataset = dataset.drop(columns=columns_to_drop)


# %%
# Pada kolom class, dataset masih memiliki tipe kategorial.
# Rubah menjadi data numerik untuk proses tahap selanjutnya.
dataset = dataset.replace(
{"label": {"DDOS_ICMP": 1, "DDOS_TCP": 2, "DDOS_UDP": 3, "NORMAL_ICMP": 4, "NORMAL_TCP": 5, "NORMAL_UDP": 6}})
print(dataset.sample(frac=0.2))

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
#K‐Fold Cross Validation

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
# Memisahkan fitur dataset (X) dan class/label dataset (y)
X = dataset.drop('label', axis=1) # or using X = dataset.drop('class', axis=1)
y = dataset['label'] # or using y = dataset['class'
# Menset model ke model yang dipilih dan mem‐fit modelnya
# model dapat diganti secara manual
model = KNeighborsClassifier(n_neighbors=3) # algortima klasifikasi Nearest Neighbor
# Menghitung dan mencetak akurasi model
kFoldValidation = KFold(10)
modelScore = cross_val_score(model, X, y, cv=kFoldValidation)
print("K‐Fold Cross Validation score: ", modelScore) # hasilnya score sebanyak K
print("Ratas KFCV score: ", np.mean(modelScore)) # menghitung ratas score dari score sejumlah K
"""
Kadang kala diperlukan ukuran error dari model sebagai kebalikan dari ukuran akurasi.
Berikut ini adalah menghitung error menggunakan MEA dan RMSE
"""
# Menghitung dan mencetak Mean Absolute Error (MAE) model
maeScore = cross_val_score(model, X, y, cv=kFoldValidation, scoring='neg_mean_absolute_error')
print("K‐Fold Cross Validation MAE: ", maeScore) # hasilnya mae sebanyak K
print("Ratas KFCV Mean Absolute Error: ", np.mean(maeScore)) # menghitung ratas mae dari mae sejumlah K
# # Menghitung dan mencetak Root Mean Square Error (RMSE) model
rmseScore = cross_val_score(model, X, y, cv=kFoldValidation, scoring='neg_mean_squared_error')
print("K‐Fold Cross Validation RMSE: ", rmseScore) # hasilnya rmse sebanyak K
print("Ratas KFCV Root Mean Square Error: ", np.mean(rmseScore)) # menghitung ratas rmse dari mae sejumlah K

# %%
#LeaveOneOut

from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
# Memisahkan fitur dataset (X) dan class/label dataset (y)
X = dataset.drop('label', axis=1) # or using X = dataset.drop('class', axis=1)
y = dataset['label'] # or using y = dataset['class'
# Menset model ke model yang dipilih dan mem‐fit modelnya
# model dapat diganti secara manual
model = KNeighborsClassifier(n_neighbors=3) # algortima klasifikasi Nearest Neighbor
# Menghitung dan mencetak akurasi model
LOO_validation = LeaveOneOut()
modelScore = cross_val_score(model, X, y, cv=LOO_validation)
print("LOOCV score: ", modelScore) # hasilnya array prediksi yaitu 1=benar dan 0=salah
print("Ratas LOOCV score: ", np.mean(modelScore)) # menghitung ratas score dari score sejumlah benar & salah
"""
Kadang kala diperlukan ukuran error dari model sebagai kebalikan dari ukuran akurasi.
Berikut ini adalah menghitung error menggunakan MEA dan RMSE """
# Menghitung dan mencetak Mean Absolute Error (MAE) model
maeScore = cross_val_score(model, X, y, cv=LOO_validation, scoring='neg_mean_absolute_error')
print("LOOCV MAE: ", maeScore) # hasilnya mae sebanyak K
print("Ratas LOOCV Mean Absolute Error: ", np.mean(maeScore)) # menghitung ratas mae dari mae sejumlah K
# # Menghitung dan mencetak Root Mean Square Error (RMSE) model
rmseScore = cross_val_score(model, X, y, cv=LOO_validation, scoring='neg_mean_squared_error')
print("LOOCV Cross Validation RMSE: ", rmseScore) # hasilnya rmse sebanyak K
print("Ratas LOOCV Root Mean Square Error: ", np.mean(rmseScore)) # menghitung ratas rmse dari mae sejumlah K

# %%
from sklearn.utils import resample  # for Bootstrap sampling
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Assuming 'dataset' is your DataFrame containing the data
# Convert dataset to array
values = dataset.values

# Configure Bootstrap
n_iterations = 10  # Number of bootstrap samples to create
n_size = int(len(dataset) * 0.5)  # Size of each bootstrap sample (50% of the dataset)

# Run Bootstrap
stats = []
for i in range(n_iterations):
    # Prepare training and test sets
    train = resample(values, replace=True, n_samples=n_size, random_state=i)  # Different random state for each iteration
    test = np.array([x for x in values if x.tolist() not in train.tolist()])

    # Fit model
    model = KNeighborsClassifier(n_neighbors=3)  # K-Nearest Neighbors classifier
    model.fit(train[:, :-1], train[:, -1])  # Train the model with training data

    # Evaluate model
    predictions = model.predict(test[:, :-1])  # Predict on the test data
    score = accuracy_score(test[:, -1], predictions)  # Calculate accuracy score
    stats.append(score)

# Print results of all bootstrap trials
print("Hasil semua percobaan bootstrap: ", stats)
print("Rata-rata Bootstrap score: ", np.mean(stats))


# %%
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import numpy as np
import time

# Convert dataset to array
values = dataset.values

# Configure Bootstrap
n_iterations = 1  # Number of bootstrap samples to create
n_size = int(len(dataset) * 0.1)  # Size of each bootstrap sample (50% of the dataset)

# Function to perform one bootstrap iteration
def bootstrap_iteration(i):
    train = resample(values, replace=True, n_samples=n_size, random_state=i)
    test = np.array([x for x in values if x.tolist() not in train.tolist()])

    model = KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree')
    model.fit(train[:, :-1], train[:, -1])

    predictions = model.predict(test[:, :-1])
    score = accuracy_score(test[:, -1], predictions)
    return score

# Start timing
start_time = time.time()

# Run Bootstrap iterations in parallel
stats = Parallel(n_jobs=-1)(delayed(bootstrap_iteration)(i) for i in range(n_iterations))

# End timing
end_time = time.time()

# Print results of all bootstrap trials
print("Hasil semua percobaan bootstrap: ", stats)
print("Rata-rata Bootstrap score: ", np.mean(stats))
print("Total time: ", end_time - start_time, "seconds")



