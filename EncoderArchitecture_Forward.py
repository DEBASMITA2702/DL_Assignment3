import Utilities_Tensor
import torch.nn as nn
from EncoderArchitecture import EncoderStack as BaseEncoderStack

'''class to add forward propagation to the encoder architecture'''
class EncoderStack(BaseEncoderStack):

    def forward(self, batchData):
        '''
            Parameters:
                batchData : data sent in batches (as a 2D tensor)
            Returns :
                modelEval : output from the current state of the encoder
                innerLayer : hidden layers representation
                model : the object of the combined architecture with updated parameters
            Function:
                Performs forward propagation in the architecture
        '''

        '''sets embedding layer'''
        embeddedBatch = self.embeddingLayer(batchData)
        embeddedBatch = self.dropout(embeddedBatch)
        model = None

        '''create the gates for LSTM'''
        if isinstance(self.model, nn.LSTM):
            modelEval, (innerLayer, model) = self.model(embeddedBatch)
            '''implement bidirectional architecture'''
            if self.biDirect:
                batchSize = model.size(1)
                model = Utilities_Tensor.resizeTensor(model, self.layersInEncoder, 2, batchSize, -1)
                model = Utilities_Tensor.reverseTensor(model)
                model = Utilities_Tensor.getMean(model)
            else:
                model = model[-1, :, :]
            model = Utilities_Tensor.increaseDimension(model)
        else:
            modelEval, innerLayer = self.model(embeddedBatch)

        '''check and implement bidirectional architecture'''
        if self.biDirect:
            batchSize = innerLayer.size(1)
            innerLayer = Utilities_Tensor.resizeTensor(innerLayer, self.layersInEncoder, 2, batchSize, -1)
            innerLayer = Utilities_Tensor.reverseTensor(innerLayer)
            innerLayer = Utilities_Tensor.getMean(innerLayer)
            '''apply attention'''
            if self.attention == 1:
                modelEval = Utilities_Tensor.addTensor(modelEval[:, :, :self.neruonsInFC], modelEval[:, :, self.neruonsInFC:])
        else:
            innerLayer = innerLayer[-1, :, :]

        innerLayer = Utilities_Tensor.increaseDimension(innerLayer)

        return modelEval, innerLayer, model