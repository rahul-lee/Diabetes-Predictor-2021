
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import joblib as jb

import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv('diabetes.csv')

dataset_X = dataset.iloc[:, [1, 4, 5, 7]].values
dataset_Y = dataset.iloc[:, 8].values

dataset_X

sc = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = sc.fit_transform(dataset_X)

dataset_scaled = pd.DataFrame(dataset_scaled)

X = dataset_scaled
Y = dataset_Y

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42, stratify=dataset['Outcome'])


# svc = SVC(kernel='linear', random_state=42)
# svc.fit(X_train, Y_train)

# K nearest neighbors Algorithm

knn = KNeighborsClassifier(n_neighbors=24, metric='minkowski', p=2)
knn.fit(X_train, Y_train)
knn.score(X_test, Y_test)
Y_pred = knn.predict(X_test)

jb.dump(knn, 'model_predict.joblib')
model = jb.load('model_predict.joblib')

# print(model.predict(sc.transform(np.array([[86, 66, 26.6, 31]]))))
