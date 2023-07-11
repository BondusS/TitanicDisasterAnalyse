import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

Dataset = pd.read_csv("train.csv")
ColumnsToDrop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
Dataset = Dataset.drop(ColumnsToDrop, axis=1)
ColumnsToFactorize = ['Sex', 'Embarked']
Dataset[ColumnsToFactorize] = Dataset[ColumnsToFactorize].apply(lambda x: pd.factorize(x)[0])
Dataset.dropna(inplace=True)
ax = Dataset.boxplot()
plt.show()
IQR = Dataset['Fare'].quantile(0.75) - Dataset['Fare'].quantile(0.25)
Dataset = Dataset[Dataset['Fare'] <= Dataset['Fare'].quantile(0.75)+1.5*IQR]
Dataset.info()
sns.heatmap(Dataset.corr(),  vmin=-1, vmax=+1, annot=True, cmap='coolwarm')
plt.show()
sns.pairplot(Dataset, height=1)
plt.show()
X_df = Dataset.drop(['Survived'], axis=1)
scaler = StandardScaler().fit(X_df)
X_st = scaler.transform(X_df)
pca = PCA()
x_pca = pca.fit_transform(X_st)
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.grid()
plt.show()
pca_new = PCA(n_components=5)
x_new = pca_new.fit_transform(X_st)
