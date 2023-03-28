import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

import numpy as np
import matplotlib.pyplot as plt

# importing a csv file using pandas library
data = pd.read_csv('dataIN/Soft Pack.csv')
print(data)

# deleting a column from the given dataset
column_name = "Date order"
data = data.drop(column_name, axis=1)
data.to_csv('dataOUT/SoftPackNew.csv')

# accessing data with loc and iloc
path="/Users/theo/PycharmProjects/SoftwarePackagesProject/"  #adaptabil in functie de device-ul de pe care ruleaza programul
df = pd.read_csv(path+'dataIN/Soft Pack.csv')
print(df.loc[(df['Continent'] == 'Europe'), ['Name', 'Continent']])

# defining and calling functions + using group functions
def count_females(dataset_path):
    df = pd.read_csv(dataset_path)
    female_count = df[df['Gender'] == 'Female'].shape[0]
    return female_count

female_count = count_females('/Users/theo/PycharmProjects/SoftwarePackagesProject/dataIN/Soft Pack.csv')
print('Number of female customers in the dataset: ', female_count)


#using scikit-learn package (clustering)
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data['Brand'] = le.fit_transform(data['Brand'])
data['Product name'] = le.fit_transform(data['Product name'])
data['Size'] = le.fit_transform(data['Size'])
data['Order value'] = le.fit_transform(data['Order value'])

X = data[['Gender', 'Brand', 'Product name', 'Size', 'Order value']]

k = 3
kmeans = KMeans(n_clusters=k, n_init=5).fit(X)

print(kmeans.cluster_centers_)
print(kmeans.labels_)

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=kmeans.labels_, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

plt.show()

