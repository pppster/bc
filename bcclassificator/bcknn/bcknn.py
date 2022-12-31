import os

import numpy as np
from numpy.linalg import norm
import pandas as pd

class BCKNN:
    def __init__(self, path: str):
        self.path = path
        self.distances = pd.DataFrame(columns=['Name', 'Label', 'Distance'])
        if os.path.isfile(path):
            self.dataset = pd.read_csv(path)
        else:
            self.dataset = pd.DataFrame()
            print(f'There is no dataset located at: {path}')

    def calculate_distances(self, new_sample: pd.DataFrame):
        new_sample_values = new_sample.to_numpy()
        for _, row in self.dataset.iterrows():
            values = row.iloc[1:-1].to_numpy()
            distance = norm(values-new_sample_values)
            name = row['Name']
            label = row['Label']
            self.distances.loc[len(self.distances)] = [name, label, distance]
        self.distances = self.distances.sort_values(by='Distance')


    def classify(self, k: int = 3):
        counter_1, counter_2 = 0, 0
        for _, row in self.distances.head(20).iterrows():
            if row['Label'] == 1:
                counter_1 += 1
            else:
                counter_2 += 1
            if counter_1 == k or counter_2 == k:
                if counter_1 > counter_2:
                    return 1
                else:
                    return 2





if __name__ == '__main__':
    path = '././dataframes/_bottle_damaged/dataframe.csv'
    knn = BCKNN(path=path)
    new_sample = knn.dataset.iloc[6, 1:-1]
    knn.calculate_distances(new_sample=new_sample)
    label = knn.classify(k=10)
    print(label)

