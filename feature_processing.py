from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def get_dataset(file):
    data = pd.read_csv(file)
    print(data.info)
    print(data.describe())
    y = data['label'].values
    data.drop(['label','id'],axis=1,inplace=True)
    data['sex'] = LabelEncoder().fit_transform(data['sex'])
    print(data['sex'])
    print(data['age'])
    data['age'] = preprocessing.scale(data['age'])
    x = data.values

    print(x)
    print(y)
    return x,y
def main():
    get_dataset('/home/hyaloid/PycharmProjects/ATEC/test.csv')

if __name__=='__main__':
    main()
