from feature_processing import *
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

dataset_x, dataset_y = get_dataset('/home/hyaloid/PycharmProjects/ATEC/test.csv')
pca = PCA(n_components=2,svd_solver='full')
dataset_x_pca = pca.fit_transform(dataset_x)

train_x, test_x, train_y, test_y = train_test_split(dataset_x_pca, dataset_y, test_size=0.35, random_state=0)
print(train_y)
print(test_y)
from sklearn.svm import SVC
svc = SVC(C=1.0, kernel='rbf', decision_function_shape='ovo')
svc.fit(train_x, train_y)
print(svc.predict(test_x))
print(svc.score(test_x, test_y))