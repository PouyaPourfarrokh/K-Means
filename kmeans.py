import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

iris = load_iris()
df = pd.DataFrame(iris.data, columns= iris.feature_names)

x = df[['sepal length (cm)','petal length (cm)']].values

model = KMeans(n_clusters=3, random_state= 42)
model.fit(x)
print(x)
centers = model.cluster_centers_
labels = model.labels_
#plt.scatter(x,labels)
plt.show()
#sns.scatterplot(x='sepel length (cm)', y= 'petal length (cm)',data= df, s=60, hue=model.labels_,palette=['green','blue','orange'])
#sns.scatterplot(x= centers[:,0],y= centers[:,1], markers='*', s=400, color='red')
#plt.savefig('clustring', dpi=100)
