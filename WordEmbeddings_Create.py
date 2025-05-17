import Utilities_Device_Trainings
import torch
import torch.nn as nn
from WordEmbeddings_Translate import WordEmbeddings as BaseWordEmbeddings

class WordEmbeddings(BaseWordEmbeddings):

    '''class to create the word embeddings'''
    def createWordEmbeddings(self, dataset, vocabulary):
        '''
            Parameters:
                dataset : dataset on which to create the embeddings
                vocabulary : vocabulary of the dataset
            Returns :
                None
            Function:
                Creates embeddings of the words
        '''
        englishDataset = dataset[:, 0]
        bengaliDataset = dataset[:, 1]

        tensorListEnglish = list()
        tensorListBengali = list()

        '''embeddings for source language'''
        language = "english"
        for one_word in englishDataset:
            tensor = self.translateWordToTensor(one_word, vocabulary, language)
            tensor = Utilities_Device_Trainings.setDevice(tensor)
            tensorListEnglish.append(tensor)
        self.englishEmbedding = nn.utils.rnn.pad_sequence(
            tensorListEnglish,
            padding_value=vocabulary.paddingIndex,
            batch_first=True
        )
        self.englishEmbedding = Utilities_Device_Trainings.setDevice(self.englishEmbedding)

        '''embeddings for target language'''
        language = "bengali"
        for one_word in bengaliDataset:
            tensor = self.translateWordToTensor(one_word, vocabulary, language)
            tensor = Utilities_Device_Trainings.setDevice(tensor)
            tensorListBengali.append(tensor)
        self.bengaliEmbedding = nn.utils.rnn.pad_sequence(
            tensorListBengali,
            padding_value=vocabulary.paddingIndex,
            batch_first=True
        )
        self.bengaliEmbedding = Utilities_Device_Trainings.setDevice(self.bengaliEmbedding)
