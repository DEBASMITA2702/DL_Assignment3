import Utilities_Tensor
import torch.nn as nn
import random

def formMatrix(dim1, dim2, dim3):
    '''
        Parameters:
            dim1 : First dimension of a tensor
            dim2 : Second dimension of the tensor
            dim3 : Third dimension of the tensor
        Returns :
            A tensor
        Function:
            Creates a tensor with all zeros following the dimensions passed as parameters
    '''
    return Utilities_Tensor.getZeroTensor(dim1, dim2, dim3)


def doTeacherForcing(decoderOutput, bengaliSequence, sequenceNumber, teacherRatio):
    '''
        Parameters:
            decoderOutput : Tensor representing the output of the previous state of the decoder
            bengaliSequence : Sequence of characters in the target language (bengali in this case)
            sequenceNumber : Index of the sequence to be considered
            teacherRatio : Threshold percentage on whether to apply teacher forching
        Returns :
            The function can return two things:
                if teacher forcing is not applied then return the output of the previous state of the decoder
                else return the actual target word
        Function:
            Performs teacher forcing in the decoder
    '''
    prediction = decoderOutput.argmax(dim=1)

    '''make a random guess and based on that decide whether or not to apply teacher forcing in the current timestamp'''
    currentGuess = random.random()
    if currentGuess < teacherRatio:
        return bengaliSequence[sequenceNumber]
    
    return prediction