import Utilities_Tensor
import Utilities_Device_Trainings
import torch.nn as nn
from DecoderArchitecture import DecoderStack as BaseDecoderStack

'''class to add forward propagation to the decoder architecture'''
class DecoderStack(BaseDecoderStack):

    def forward(self, batchData, encoderOutput, innerLayer, model):
        '''
            Parameters:
                batchData : data sent in batches (as a 2D tensor)
                encoderOutput : output from the encoder (on which the decoder will work)
                innerLayer : hidden layers representation
                model : the object of the combined architecture on which the decoder is working
            Returns :
                predictions : predicted outputs from the decoder
                innerLayer : hidden layers representation
                model : the object of the combined architecture with updated parameters
                finalAttentionWeights : updated attention weights
            Function:
                Performs forward propagation in the architecture
        '''

        '''sets batch size and embedding layer'''
        batchData = Utilities_Tensor.increaseDimension(batchData)
        embeddedBatch = self.embeddingLayer(batchData)
        embeddingLayer = self.dropout(embeddedBatch)

        '''declare the attention matrix'''
        finalAttentionWeights = None

        '''appply attention and calculate the weights'''
        if self.attention == 1:
            finalOutputFromEncoderBlock = self.neuronsInAttentionFC(encoderOutput)
            finalHiddenLayer = innerLayer[-1:]
            attentionValues = Utilities_Tensor.mutiplyTensors(
                Utilities_Tensor.reorderDimensions(finalOutputFromEncoderBlock, 1, 0, 2),
                Utilities_Tensor.reorderDimensions(finalHiddenLayer, 1, 2, 0)
            )
            attentionValues = Utilities_Tensor.reorderDimensions(attentionValues, 2, 0, 1)
            finalAttentionWeights = Utilities_Device_Trainings.setOutputFunction(attentionValues)
            attentionIntoDecoder = Utilities_Tensor.mutiplyTensors(
                Utilities_Tensor.reorderDimensions(finalAttentionWeights, 1, 0, 2),
                Utilities_Tensor.reorderDimensions(encoderOutput, 1, 0, 2)
            )
            attentionIntoDecoder = Utilities_Tensor.reorderDimensions(attentionIntoDecoder, 1, 0, 2)

        '''check and apply attention'''
        if self.attention == 0:
            '''apply forget gate for LSTM'''
            if isinstance(self.model, nn.LSTM):
                modelEval, (innerLayer, model) = self.model(embeddingLayer, (innerLayer, model))
            else:
                modelEval, innerLayer = self.model(embeddingLayer, innerLayer)
            '''get decoder outputs by passing through the fully connected layer'''
            predictions = self.fullyConnectedLayer(modelEval)
        else:
            '''apply forget gate for LSTM'''
            concatenatedInput = Utilities_Tensor.concatenateTensor(embeddingLayer, attentionIntoDecoder, 2)
            if isinstance(self.model, nn.LSTM):
                modelEval, (innerLayer, model) = self.model(concatenatedInput, (innerLayer, model))
            else:
                modelEval, innerLayer = self.model(concatenatedInput, innerLayer)
            concatenatedInput = Utilities_Tensor.concatenateTensor(modelEval, attentionIntoDecoder, 2)
            '''get decoder outputs by passing through the fully connected layer'''
            predictions = self.fullyConnectedLayer(concatenatedInput)

        predictions = Utilities_Tensor.decreaseDimension(predictions)

        if self.attention == 1:
            finalAttentionWeights = Utilities_Tensor.decreaseDimension(finalAttentionWeights)

        return predictions, innerLayer, model, finalAttentionWeights