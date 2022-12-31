import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


class BCNaiveBayes:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = pd.read_csv(self.dataset_path)
        self.model = GaussianNB()
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None


    def prepare_training(self, test_size: float = 0.2, random_state: int = 42, shuffle: bool = True):
        x = self.dataset.iloc[:, 1:-1]
        y = self.dataset.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state, shuffle=shuffle)
        self.x_train = x_train.to_numpy()
        self.x_test = x_test.to_numpy()
        self.y_train = y_train.to_numpy()
        self.y_test = y_test.to_numpy()


    def train(self):
        self.model = self.model.fit(self.x_train, self.y_train)
        print('Model is trained')

    def test(self):
        self.y_pred = self.model.predict(self.x_test)

    def show_results(self):
        print(f'Total: {self.y_test.shape[0]} \nMislabeled: {(self.y_test != self.y_pred).sum()}')



dataset_path = '../../dataframes/_bottle_damaged/dataframe.csv'
classifier = BCNaiveBayes(dataset_path=dataset_path)
classifier.prepare_training()
classifier.train()
classifier.test()
classifier.show_results()