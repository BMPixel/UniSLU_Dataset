from utils.datasets import *
from utils.preprocess import *

data_path2dataset = {}
for data_path in ALL_DATA_PATH:
    data_path2dataset[data_path] = ProcessedDataset(data_path, section='train')
