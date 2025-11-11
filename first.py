# Cell 1: Install Libraries
!pip install pandas numpy matplotlib seaborn scikit-learn

# Cell 2: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Cell 3: Load Dataset
# Load the Dataset into the pandas data frame
# Note: seaborn has the iris dataset built-in
iris = sns.load_dataset('iris')

# Cell 4: Display Initial Statistics
print("--- 4. Initial Statistics ---")
print("First 5 rows of the dataset:")
print(iris.head())
print("\nDataset shape:")
print(iris.shape)
print("\nDataset information:")
iris.info()
print("\nDescriptive statistics:")
print(iris.describe())
print("\nClass distribution:")
print(iris['species'].value_counts())
print("-" * 30)

# Cell 5: Check for Missing Values and Duplicates
print("--- 5. Missing Values and Duplicates ---")
print("Missing values in each column:")
print(iris.isnull().sum())
print("\nNumber of duplicate rows:")
print(iris.duplicated().sum())
print("-" * 30)

# Cell 6: Identify Outliers (IQR Method)
print("--- 6. Outlier Identification ---")
# Define numeric columns to check
numeric_columns = iris.select_dtypes(include=[np.number]).columns

# Calculate Q1, Q3, and IQR for each numeric column
Q1 = iris[numeric_columns].quantile(0.25)
Q3 = iris[numeric_columns].quantile(0.75)
IQR = Q3 - Q1

# Identify outliers
outliers = ((iris[numeric_columns] < (Q1 - 1.5 * IQR)) | (iris[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)

print("Number of rows with outliers:", outliers.sum())
print("Outlier rows:")
print(iris[outliers])
print("-" * 30)

# Cell 7: Apply Min-Max Scaling
print("--- 7. Min-Max Scaling ---")
# Apply data transformations Min-Max scaling
scaler = MinMaxScaler()
iris_scaled = iris.copy()
iris_scaled[numeric_columns] = scaler.fit_transform(iris[numeric_columns])

print("After Min-Max scaling - Descriptive statistics:")
print(iris_scaled[numeric_columns].describe())
print("-" * 30)

# Cell 8: Apply Label Encoding
print("--- 8. Label Encoding ---")
# Turn categorical variables into quantitative variables
# Method: Label Encoding
label_encoder = LabelEncoder()
iris_encoded = iris.copy() # Using the original 'iris' df for clarity
iris_encoded['species_encoded'] = label_encoder.fit_transform(iris_encoded['species'])

print("After Label Encoding (showing original and new columns):")
print(iris_encoded[['species', 'species_encoded']].head(10))
print("-" * 30)