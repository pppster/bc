import matplotlib.pyplot as plt
from typing import Optional
from bcdf.utils.utils import get_area, get_extend, get_perimeter, get_solidity, get_centroid_x, get_num_corners, get_aspect_ratio
from bci.core.bcimage import BCImage
from bci.utils.bcimageenum import BCImageEnum, IMAGE, MASK, NOBACKGROUND
import pandas as pd
from pathlib import Path
import os
from sklearn.preprocessing import MinMaxScaler
from category_encoders import OrdinalEncoder
from bcdf.utils import utils


class BCDataframe:

    def __init__(self, path: str, image_directory: str):
        self.path = path
        self.image_directory = utils.just_image_names(image_directory)
        if os.path.isfile(path):
            self.dataframe = self.load_dataframe()
        else:
            self.create_labeled_dataframe()

    def load_dataframe(self) -> pd.DataFrame:
        dataframe = pd.read_csv(self.path)
        return dataframe

    def create_labeled_dataframe(self):
        self.dataframe = utils._create_labeled_dataframe(dir=self.image_directory)

    def update_dataframe(self):
        for image in self.image_directory:
            label = image.split('\\')[-2]
            filename = image.split('\\')[-1]
            name = filename[:filename.rfind('.')]
            if not any(self.dataframe['Name'].values == name):
                self.dataframe.loc[len(self.dataframe)] = [name, label]

    def save_dataframe(self):
        filepath = Path(self.path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.dataframe.to_csv(filepath, index=False)

    def add_feature(self, column_name: str, function):
        self.dataframe.insert(1, column=column_name, value=0)
        for image_path in self.image_directory:
            img = BCImage(path=image_path)
            feature_value = function(img)
            self.dataframe.loc[(self.dataframe.Name == img.name), column_name] = feature_value

    def encode_label(self):
        encoder = OrdinalEncoder(cols=['Label'], return_df=True)
        self.dataframe = encoder.fit_transform(self.dataframe)

    def show_scatter_matrix(self):
        axes = pd.plotting.scatter_matrix(self.dataframe.iloc[:, 1:-1], c=self.dataframe.Label)
        for ax in axes.flatten():
            ax.yaxis.label.set_rotation(0)
            ax.yaxis.label.set_ha('right')
        plt.show()


    def create_test_dataframe(self, test_image: Optional[str] = None):
        columns = self.dataframe.columns
        test_dataframe = pd.DataFrame(columns=columns)
        img = BCImage(path=test_image)
        new_row = {}

        for column in test_dataframe:
            if column == 'Name':
                new_row[column] = img.name
            elif column == 'Label':
                new_row[column] = img.label
            else:
                function_name = f'get_{column}'
                func = getattr(utils, function_name)
                new_row[column] = func(img)

        new_row = pd.DataFrame([new_row])
        test_dataframe = pd.concat([test_dataframe, new_row], ignore_index=True)
        print(test_dataframe.head())
        return test_dataframe

    def normalize(self):
        values = self.dataframe.iloc[:, 1:-1].values  # returns a numpy array
        min_max_scaler = MinMaxScaler()
        values = min_max_scaler.fit_transform(values)
        self.dataframe.iloc[:, 1:-1] = values

if __name__ == '__main__':
    path = f'../../dataframes/_label_damaged/dataframe.csv'
    dir = '../../images/_label_damaged/*/*'

    bcd = BCDataframe(path=path, image_directory=dir)
    bcd.add_feature(column_name='area', function=get_area)
    bcd.add_feature(column_name='extend', function=get_extend)
    bcd.add_feature(column_name='perimeter', function=get_perimeter)
    bcd.add_feature(column_name='solidity', function=get_solidity)
    bcd.add_feature(column_name='centroid_x', function=get_centroid_x)
    bcd.add_feature(column_name='num_corners', function=get_num_corners)
    bcd.add_feature(column_name='aspect_ratio', function=get_aspect_ratio)
    bcd.normalize()
    bcd.encode_label()
    bcd.save_dataframe()
    bcd.show_scatter_matrix()






