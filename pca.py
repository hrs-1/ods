import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# define the column names
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv(url, names=features)

# Separate features and target
x = df.loc[:, ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = df.loc[:,['class']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents, 
                           columns = ['principal_component_1', 'principal_component_2'])
finalDf = pd.concat([principalDf, df[['class']]], axis = 1)

# Plot the 2 component PCA
plt.figure(figsize=(8, 6)) # Added figsize for a better plot
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2 Component PCA')

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
# Note: The line below is necessary to define the 'colors' variable used in the loop.
colors = ['r', 'g', 'b'] 

for target, color in zip(targets, colors):
    indicesToKeep = finalDf['class'] == target
    plt.scatter(finalDf.loc[indicesToKeep, 'principal_component_1'],
                finalDf.loc[indicesToKeep, 'principal_component_2'],
                c = color,
                s = 50)
plt.legend(targets)
plt.grid()
plt.show()