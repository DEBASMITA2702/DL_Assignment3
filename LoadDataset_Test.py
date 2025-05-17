import pandas as pd
import os
from LoadDataset_Train import loadDataset

def loadTestDataset(self, root, lang):
    '''
        Parameters:
            root : path of the dataset
            lang : language which is chosen (taken from the path itself)
        Returns :
            None
        Function:
            Loads test dataset
    '''
    test_path = os.path.join(root, f"{lang}.translit.sampled.test.tsv")
    test_df = pd.read_csv(test_path, sep="\t", header=None, dtype=str)
    test_df = test_df.dropna(subset=[0, 1])
    self.test_dataframe = test_df               
    self.test_dataset = test_df[[1, 0]].values    

'''class to load dataset'''
class DatasetLoad:
    pass

'''attach methods to DatasetLoad '''
DatasetLoad.loadDataset = loadDataset
DatasetLoad.loadTestDataset = loadTestDataset