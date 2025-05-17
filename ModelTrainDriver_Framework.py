import torch
from ModelTrainDriver_Initialize import Model, device
import Utilities_Device_Trainings
from EncoderArchitecture_Forward import EncoderStack
from DecoderArchitecture_Forward import DecoderStack
from CombinedModelArchitecture_Stack import EncoderDecoderStack
from RunTrainer import Trainer

def createModelFramework(self, modelType, embeddingSize, neruonsInFC, layersInEncoder, layersInDecoder, dropout, bidirectional, learningRate, epochs, batchSize):
    '''
        Parameters:
            modelType : type of cell (RNN, LSTM, GRU)
            embeddingSize : size of the embeddings
            neruonsInFC : number of neurons in the fully connected layer
            layersInEncoder : number of layers in the encoder
            layersInDecoder : number of layers in the decoder
            dropout : probability of dropout
            bidirectional : variable indicating whether to apply bidirectional flow or not
            learningRate : learning rate of the model
            epochs : number of epochs to run
            batchSize : batch size used
        Returns :
            None
        Function:
            Runs the encoder-decoder architecture on the data passed
    '''

    '''create encoder object'''
    paramList = [modelType, self.encoderInputSize, embeddingSize, neruonsInFC, layersInEncoder, dropout, bidirectional, self.attention]
    self.encoderFramework = EncoderStack(paramList)
    self.encoderFramework = Utilities_Device_Trainings.setDevice(self.encoderFramework)

    '''create decoder object'''
    paramList = [modelType, self.decoderInputSize, embeddingSize, neruonsInFC, self.outputWordSize, layersInDecoder, dropout, self.attention]
    self.decoderFramework = DecoderStack(paramList)
    self.decoderFramework = Utilities_Device_Trainings.setDevice(self.decoderFramework)

    '''create the combined architecture'''
    paramList = [self.encoderFramework, self.decoderFramework, self.attention]
    self.framework = EncoderDecoderStack(paramList)
    self.framework = Utilities_Device_Trainings.setDevice(self.framework)
    
    '''
        check if this is a train.py call.
        If yes then train the model and return the trained model
    '''
    if self.trainPy == 1:
        paramList = [self.framework, learningRate, self.trainEmbeddedDataLoader, self.valEmbeddedDataLoader, epochs, batchSize, self.paddingIndex]
        framework = Trainer.runModelTrainer(paramList, self.trainPy, logging=1)
        return framework

    else:
        '''if testing is done then no need of training (load the best model that is saved)'''
        if self.test == 0:
            paramList = [self.framework, learningRate, self.trainEmbeddedDataLoader, self.valEmbeddedDataLoader, epochs, batchSize, self.paddingIndex]
            Trainer.runModelTrainer(paramList, logging=1)
        else:
            '''Train the model during test mode with the best configuration'''
            paramList = [modelType, self.encoderInputSize, embeddingSize, neruonsInFC, layersInEncoder, dropout, bidirectional, self.attention]
            self.encoderFramework = EncoderStack(paramList)
            self.encoderFramework = Utilities_Device_Trainings.setDevice(self.encoderFramework)

            paramList = [modelType, self.decoderInputSize, embeddingSize, neruonsInFC, self.outputWordSize, layersInDecoder, dropout, self.attention]
            self.decoderFramework = DecoderStack(paramList)
            self.decoderFramework = Utilities_Device_Trainings.setDevice(self.decoderFramework)

            paramList = [self.encoderFramework, self.decoderFramework, self.attention]
            self.framework = EncoderDecoderStack(paramList)
            self.framework = Utilities_Device_Trainings.setDevice(self.framework)

            paramList = [self.framework, learningRate, self.trainEmbeddedDataLoader, self.valEmbeddedDataLoader, epochs, batchSize, self.paddingIndex]
            self.framework = Trainer.runModelTrainer(paramList, logging=0)
            

'''attach the split method to Model'''
Model.createModelFramework = createModelFramework