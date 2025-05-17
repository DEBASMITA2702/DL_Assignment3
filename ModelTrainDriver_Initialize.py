import torch
from EncoderArchitecture import EncoderStack
from DecoderArchitecture import DecoderStack
from CombinedModelArchitecture_Stack import EncoderDecoderStack
from RunTrainer import Trainer

'''setting device to cpu to load the saved model during testing'''
device = torch.device('cpu')

'''class to drive the steps of training the model'''
class Model:
    
    '''constructor to intialize the class parameters'''
    def __init__(self, vocabulary, trainEmbeddedDataLoader, valEmbeddedDataLoader, test=0, attention=0, trainPy=0):
        '''
            Parameters:
                vocabulary : vocabulary of the dataset
                trainEmbeddedDataLoader : training data
                valEmbeddedDataLoader : validation data
                test : variable indicating whether to do test or not
                attention : variable indicating whether to apply attention or not
                root : path of the dataset
                trainPy : variable indicating whether to this is train.py call or not
            Returns :
                None
            Function:
                Sets class parameters
        '''
        self.paddingIndex = vocabulary.paddingIndex
        self.encoderInputSize = vocabulary.vocabularySizeForEnglish
        self.decoderInputSize = vocabulary.vocabularySizeForBengali
        self.outputWordSize = vocabulary.vocabularySizeForBengali
        self.trainEmbeddedDataLoader = trainEmbeddedDataLoader
        self.valEmbeddedDataLoader = valEmbeddedDataLoader
        self.test = test
        self.attention = attention
        self.trainPy = trainPy