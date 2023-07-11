import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

Dataset = pd.read_csv("train.csv")
ColumnsToDrop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
Dataset = Dataset.drop(ColumnsToDrop, axis=1)
ColumnsToFactorize = ['Sex', 'Embarked']
Dataset[ColumnsToFactorize] = Dataset[ColumnsToFactorize].apply(lambda x: pd.factorize(x)[0])
Dataset.dropna(inplace=True)
IQR = Dataset['Fare'].quantile(0.75) - Dataset['Fare'].quantile(0.25)
Dataset = Dataset[Dataset['Fare'] <= Dataset['Fare'].quantile(0.75)+1.5*IQR]
X_df = Dataset.drop(['Survived'], axis=1)
X_np = X_df.to_numpy()
scaler = StandardScaler().fit(X_np)
X_st = scaler.transform(X_np)
pca = PCA()
x_pca = pca.fit_transform(X_st)
pca_new = PCA(n_components=5)
x_new = pca_new.fit_transform(X_st)
Y_df = Dataset['Survived']
Y_np = Y_df.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x_new, Y_np, stratify=Dataset['Survived'])
print('GradientBoostingClassifier')
Model = GradientBoostingClassifier().fit(x_train, y_train)
print('Score правильных ответов на обучающей выборке ', Model.score(x_train, y_train))
print('Score правильных ответов на тестовой выборке ', Model.score(x_test, y_test))
print('Подборка лучших параметров...')
params = {'n_estimators': [n for n in range(100, 1001, 100)],
          'max_depth': [n for n in range(1, 11, 1)]}
grid = GridSearchCV(estimator=Model, param_grid=params)
grid.fit(x_train, y_train)
print('Наилучший score при подборке наиболее подходящих параметров ', grid.best_score_)
print('Best n_estimators', grid.best_estimator_.n_estimators)
print('Best max_depth', grid.best_estimator_.max_depth)
BestModel = GradientBoostingClassifier(n_estimators=grid.best_estimator_.n_estimators,
                                       max_depth=grid.best_estimator_.max_depth).fit(x_train, y_train)
print('Доля правильных ответов на обучающей выборке ', BestModel.score(x_train, y_train))
print('Доля правильных ответов на тестовой выборке ', BestModel.score(x_test, y_test))
print('SVC')
Model = SVC(kernel='rbf').fit(x_train, y_train)
print('Score правильных ответов на обучающей выборке ', Model.score(x_train, y_train))
print('Score правильных ответов на тестовой выборке ', Model.score(x_test, y_test))
print('Подборка лучших параметров...')
params = {'gamma': [n*0.1 for n in range(1, 11, 1)],
          'C': [i*0.1 for i in range(1, 21, 1)]}
grid = GridSearchCV(estimator=Model, param_grid=params)
grid.fit(x_train, y_train)
print('Наилучший score при подборке наиболее подходящих параметров ', grid.best_score_)
print('Best gamma', grid.best_estimator_.gamma)
print('Best C', grid.best_estimator_.C)
BestModel = SVC(gamma=grid.best_estimator_.gamma,
                C=grid.best_estimator_.C,
                kernel='rbf').fit(x_train, y_train)
print('Доля правильных ответов на обучающей выборке ', BestModel.score(x_train, y_train))
print('Доля правильных ответов на тестовой выборке ', BestModel.score(x_test, y_test))
