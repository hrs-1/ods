import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load the iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv(url, header=None, names=column_names)

# Encode target variable (species) to numeric values
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Split the dataset into features and target variable
X = df.iloc[:, :-1] # All rows, all columns except the last one
y = df['species'] # Target variable

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize kNN classifier with k=3 (you can change k as needed)
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier on the training data
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Note: The TP/FP/TN/FN extraction below is for binary classification (2x2 matrix).
# Since Iris is a multi-class problem (3x3 matrix), these specific values
# are not fully representative. The metrics (precision, recall, etc.)
# using average='weighted' are the correct way to evaluate.
# TP = cm[1, 1] # True Positive
# FN = cm[1, 0] # False Negative
# FP = cm[0, 1] # False Positive
# TN = cm[0, 0] # True Negative

# Compute other metrics
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print results
print("Confusion Matrix:")
print(cm)
# print(f"True Positives (TP): {TP}")
# print(f"False Positives (FP): {FP}")
# print(f"True Negatives (TN): {TN}")
# print(f"False Negatives (FN): {FN}")
print(f"Accuracy: {accuracy}")
print(f"Error Rate: {error_rate}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")