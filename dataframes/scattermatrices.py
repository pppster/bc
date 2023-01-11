from bcdf.core.bcdataframe import BCDataframe
import os

path_bottle_damaged = f'./_bottle_damaged/dataframe.csv'
path_label_damaged = f'./_label_damaged/dataframe.csv'
path_open_close = f'./_open_closed/dataframe.csv'
paths = [
    path_bottle_damaged,
    path_label_damaged,
    path_open_close
]

dir_bottle_damaged = '../images/_bottle_damaged/*/*'
dir_label_damaged = '../images/_label_damaged/*/*'
dir_open_close = '../images/_open_close/*/*'

dirs = [
    dir_bottle_damaged,
    dir_label_damaged,
    dir_open_close
]

for path, dir in zip(paths, dirs):
    df = BCDataframe(path=path, image_directory=dir)
    df.show_scatter_matrix()
