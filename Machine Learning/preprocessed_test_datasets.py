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
dataset = pd.read_csv('TEST-DATA.csv')

# Kelompokkan data berdasarkan label
grouped_data = dataset.groupby('label')

# Tampilkan 5 data pertama dari setiap label
for label, label_data in grouped_data:
  print(f"\nLabel: {label}")
  print(label_data.head().to_string(index=False))

# %%
# Check the features
dataset.info()


# %%
# menampilkan informasi statistik dataset
dataset.describe()

# %%
#drop column dikarenakan data setiap fitur tidak mengelompok atau tersebar jauh dari pusat kelompok (mean).
columns_to_drop = ['tos', 'flags', 'offset', 'code_icmp', 'rx_error_ave', 'rx_dropped_ave', 'tx_error_ave', 'tx_dropped_ave']
dataset = dataset.drop(columns=columns_to_drop)

# %%
#drop column dikarenakan bertipe object
columns_to_drop = ['src_ip', 'dst_ip']
dataset = dataset.drop(columns=columns_to_drop)


# %%
# Check the dataset if there's NaN value
print(dataset.isna().values.any())

# %%
# Mengganti kolom label string
iris = dataset.replace(
{"label": {"DDOS_ICMP": 1, "DDOS_TCP": 2, "DDOS_UDP": 3, "NORMAL_ICMP": 4, "NORMAL_TCP": 5, "NORMAL_UDP": 6}})
# menampilkan heatmap (correlation matrix)
plt.figure(figsize=(8, 6))  
sns.heatmap(iris.corr(numeric_only=True), annot=True)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.title("Correlation on DDOS attack label")
plt.tight_layout()
plt.show()

# %%

# Calculate the correlation matrix directly
correlation_matrix = iris.corr()

# Convert correlation matrix to a flat Series
corr_values = correlation_matrix.stack()

# Sort by absolute correlation (descending order)
sorted_corr = corr_values.abs().sort_values(ascending=False)

# Print the most correlated features
print("From top to the least Correlated Features:")
for feature, correlation in sorted_corr[:50].items():
    print(f"{feature}: {correlation:.3f}")


# %%
# Pilih kolom 
kolom= ['type_icmp', 'csum_icmp', 'rx_bytes_ave', 'ttl','label']

# Buat matriks plotting
plt.figure(figsize=(8, 8))
ax = sns.pairplot(dataset[kolom], hue='label')

# Tampilkan plot
plt.show()

# %%
# Save preprocessed data to a CSV file
dataset.to_csv('preprocessed_test_datasets.csv', index=False)
print("Preprocessed data saved to preprocessed_test_datasets.csv")

# %%
print(dataset.columns)



