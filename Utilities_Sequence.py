import torch
import numpy as np

def runDecoderWithNoTeacherForcing(framework, input, output, neruonsInFC):
    '''
        Parameters:
            framework : the model on which to run decoder without teacher forcing
            input : input to the decoder
            output : output from the encoder
            neruonsInFC : number of neurons in the fully connected layer
        Returns :
            modelEval : output after running the encoder-decoder architecture
        Function:
            Performs decoder run with no teacher forcing
    '''
    modelEval, model = framework(input, output, neruonsInFC, teacherRatio=0.0)
    return modelEval, model


def getBatchFloorValue(x, y):
    '''
        Parameters:
           x,y : Values whose floor to calculate
        Returns :
            an integer
        Function:
            Calculates and returns floor value
    '''
    floorValue = np.floor(x / y)
    return int(floorValue)