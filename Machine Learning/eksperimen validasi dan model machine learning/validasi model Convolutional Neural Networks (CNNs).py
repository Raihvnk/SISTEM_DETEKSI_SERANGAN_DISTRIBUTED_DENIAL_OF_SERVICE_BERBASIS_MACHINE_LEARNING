# %%
import pandas as pd

# Import dataset
dataset = pd.read_csv('preprocessed_train_datasets.csv')

# %%
#Convolutional Neural Networks (CNNs) subsampling
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Memisahkan fitur dataset (X) dan label dataset (y)
X = dataset.drop('label', axis=1).values
y = dataset['label'].values

# Normalisasi fitur
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Reshape data untuk Conv1D (CNN untuk data tabular biasanya memerlukan reshape)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Membuat model CNN
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # Gunakan 'softmax' jika lebih dari 2 kelas

# Mengkompilasi model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # Gunakan 'categorical_crossentropy' jika lebih dari 2 kelas

# Melatih model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# Menghitung dan mencetak akurasi model
loss, accuracy = model.evaluate(X_test, y_test)
print("Random Subsampling CNN score: ", accuracy)


# %%
#Convolutional Neural Networks (CNNs) dengan metode Holdout
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Mengubah dataset menjadi format yang sesuai untuk CNN (misalnya, gambar 2D dengan 1 saluran warna)
# Misalkan data Anda adalah data gambar, jadi Anda perlu mereshape X menjadi format (n_samples, height, width, channels)
# Jika tidak, Anda perlu memastikan data Anda dalam format yang benar
# X = X.reshape((X.shape[0], height, width, channels)) 

# Memisahkan fitur dataset (X) dan label dataset (y)
X = dataset.drop('label', axis=1).values
y = dataset['label'].values

# Mengubah label menjadi format one-hot encoding jika diperlukan
y = to_categorical(y)

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Mengubah data menjadi format float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Menset model CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1:])))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax')) # jumlah unit output sama dengan jumlah kelas

# Kompilasi model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Melatih (fit) model menggunakan X_train, y_train data
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Menghitung dan mencetak akurasi model dengan metode Holdout
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Holdout score: ", test_acc)


# %%
#CNNs with K-Fold Cross Validation:
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Memisahkan fitur dataset (X) dan class/label dataset (y)
X = dataset.drop('label', axis=1).values
y = dataset['label'].values

# Standarisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape X untuk CNN (misalnya, jika data 1D, reshape menjadi [samples, timesteps, features])
X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Definisikan model CNN
def create_cnn_model():
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(X_scaled.shape[1], 1)))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# K-Fold Cross Validation
kFold = KFold(n_splits=10, shuffle=True, random_state=42)

cv_scores = []
mae_scores = []
rmse_scores = []

for train_index, val_index in kFold.split(X_scaled):
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    model = create_cnn_model()
    
    # Latih model
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    # Evaluasi model
    scores = model.evaluate(X_val, y_val, verbose=0)
    cv_scores.append(scores[1])  # Accuracy
    
    y_pred = model.predict(X_val)
    mae = np.mean(np.abs(y_val - y_pred))
    rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
    
    mae_scores.append(mae)
    rmse_scores.append(rmse)

print("K-Fold Cross Validation Accuracy: ", cv_scores)
print("Rata-rata KFCV Accuracy: ", np.mean(cv_scores))
print("K-Fold Cross Validation MAE: ", mae_scores)
print("Rata-rata KFCV Mean Absolute Error: ", np.mean(mae_scores))
print("K-Fold Cross Validation RMSE: ", rmse_scores)
print("Rata-rata KFCV Root Mean Square Error: ", np.mean(rmse_scores))



