import pandas as pd
import os

def loadDataset(self, root, lang):
    train_path = os.path.join(root, f"{lang}.translit.sampled.train.tsv")
    val_path   = os.path.join(root, f"{lang}.translit.sampled.dev.tsv")

    train_df = pd.read_csv(train_path, sep="\t", header=None, dtype=str)
    val_df = pd.read_csv(val_path, sep="\t", header=None, dtype=str)

    ''' Drop rows where either source or target is missing'''
    train_df = train_df.dropna(subset=[0, 1])
    val_df = val_df.dropna(subset=[0, 1])

    ''' Select only the first two columns'''
    self.train_dataset = train_df[[1, 0]].values  
    self.val_dataset = val_df[[1, 0]].values