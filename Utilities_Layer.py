import torch.nn as nn

def createEmbeddingLayer(layerSize1, layerSize2):
    '''
        Parameters:
            layerSize1,layerSize2 : size of the layers to produce the embedding layer
        Returns :
            an object of the embedding layer
        Function:
            Creates embedding layer
    '''
    return nn.Embedding(layerSize1, layerSize2)


def createLinearLayer(neuronsInLayer1, neuronsInLayer2, bias):
    '''
        Parameters:
            neuronsInLayer1,neuronsInLayer2 : number of neurons to produce the linear layer
            bias : variable indicating whether to apply bias or not
        Returns :
            an object of the linear layer
        Function:
            Creates linear layer
    '''
    return nn.Linear(neuronsInLayer1, neuronsInLayer2, bias=bias)


def createDropoutLayer(percentage):
    '''
        Parameters:
            percentage : percentage of dropout to be applied
        Returns :
            an object of the dropout layer
        Function:
            Creates dropout layer
    '''
    return nn.Dropout(percentage)