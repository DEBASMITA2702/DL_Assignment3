import Utilities_Layer
import torch.nn as nn

'''class to represent the encoder architecture'''
class EncoderStack(nn.Module):

    '''constructor to intialize the class parameters'''
    def __init__(self, argList):
        '''inherit the constructor of the parent class'''
        super(EncoderStack, self).__init__()
        '''set all the class parameters based on the arguments passed'''
        modelType = argList[0]
        encoderInputSize = argList[1]
        embeddingSize = argList[2]
        neruonsInFC = argList[3]
        layersInEncoder = argList[4]
        dropout = argList[5]
        biDirectional = argList[6]
        attention = argList[7]

        self.neruonsInFC = neruonsInFC
        self.layersInEncoder = layersInEncoder
        if biDirectional == "YES":
            self.biDirect = True
        else:
            self.biDirect = False
        self.attention = attention

        '''select the cell type based on the value passed in argument'''
        model_dict = {"LSTM": nn.LSTM, "GRU": nn.GRU, "RNN": nn.RNN}
        modelObj = model_dict.get(modelType)

        '''do not apply dropout if only one layer is present'''
        if self.layersInEncoder == 1:
            self.dropout = Utilities_Layer.createDropoutLayer(0.0)
            self.model = modelObj(embeddingSize, self.neruonsInFC, self.layersInEncoder, dropout=0.0, bidirectional=self.biDirect)
        else:
            self.dropout = Utilities_Layer.createDropoutLayer(dropout)
            self.model = modelObj(embeddingSize, self.neruonsInFC, self.layersInEncoder, dropout=dropout, bidirectional=self.biDirect)

        '''create ambedding layer'''
        self.embeddingLayer = Utilities_Layer.createEmbeddingLayer(encoderInputSize, embeddingSize)