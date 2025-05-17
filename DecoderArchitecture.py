import Utilities_Layer
import torch.nn as nn

'''class to represent the decoder architecture'''
class DecoderStack(nn.Module):

    '''constructor to intialize the class parameters'''
    def __init__(self, argList):
        '''inherit the constructor of the parent class'''
        super(DecoderStack, self).__init__()
        '''set all the class parameters based on the arguments passed'''
        modelType = argList[0]
        decoderInputSize = argList[1]
        embeddingSize = argList[2]
        neruonsInFC = argList[3]
        outputWordSize = argList[4]
        layersInDecoder = argList[5]
        dropout = argList[6]
        attention = argList[7]

        self.modelType = modelType
        self.layersInDecoder = layersInDecoder
        self.outputWordSize = outputWordSize
        self.attention = attention
        
        '''select the cell type based on the value passed in argument'''
        modelDict = {"LSTM": nn.LSTM, "GRU": nn.GRU, "RNN": nn.RNN}
        modelObj = modelDict.get(modelType)

        '''apply attention'''
        if self.attention == 0:
            '''do not apply dropout if only one layer is present'''
            if layersInDecoder == 1:
                self.dropout = Utilities_Layer.createDropoutLayer(0.0)
                self.model = modelObj(embeddingSize, neruonsInFC, layersInDecoder, dropout=0.0)
            else:
                self.dropout = Utilities_Layer.createDropoutLayer(dropout)
                self.model = modelObj(embeddingSize, neruonsInFC, layersInDecoder, dropout=dropout)
            self.fullyConnectedLayer = nn.Linear(neruonsInFC, outputWordSize)
        else:
            '''do not apply dropout if only one layer is present'''
            if layersInDecoder == 1:
                self.dropout = Utilities_Layer.createDropoutLayer(0.0)
                self.model = modelObj(embeddingSize + neruonsInFC, neruonsInFC, layersInDecoder, dropout=0.0)
            else:
                self.dropout = Utilities_Layer.createDropoutLayer(dropout)
                self.model = modelObj(embeddingSize + neruonsInFC, neruonsInFC, layersInDecoder, dropout=dropout)
            self.fullyConnectedLayer = nn.Linear(neruonsInFC * 2, outputWordSize)
        
        '''create ambedding and linear layer'''
        self.embeddingLayer = Utilities_Layer.createEmbeddingLayer(decoderInputSize, embeddingSize)
        self.neuronsInAttentionFC = Utilities_Layer.createLinearLayer(neruonsInFC, neruonsInFC, False)