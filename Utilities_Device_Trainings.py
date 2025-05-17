import torch
from torch import optim
import torch.nn as nn

def setDevice(objToSet):
    '''
        Parameters:
            objToSet : object on which to set the device
        Returns :
            objToSet : the same object after the device is set on it
        Function:
            Sets the device as cpu or gpu based on availability
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    objToSet = objToSet.to(device)
    return objToSet


def setOptimizer(framework, learningRate):
    '''
        Parameters:
            framework : the model on which to set the opotimizer
            learningRate : learning rate to be applied
        Returns :
            An object of the optimizer
        Function:
            Sets the optimizer
    '''
    return optim.Adam(framework.parameters(), lr=learningRate)


def setLossFunction():
    '''
        Parameters:
            None
        Returns :
            An object of the loss function
        Function:
            Sets the loss function
    '''
    return nn.CrossEntropyLoss()


def setOutputFunction(layer):
    '''
        Parameters:
            layer : layer on which to apply softmax
        Returns :
            An object of the softmax function
        Function:
            Sets the output function as softmax
    '''
    return nn.functional.softmax(layer, dim=2)


def clipGradient(framework):
    '''
        Parameters:
            framework : the model on which to do gradient clipping
        Returns :
            framework : the same model object after gradient clipping is done
        Function:
            Performs gradient clipping
    '''
    torch.nn.utils.clip_grad_norm_(framework.parameters(), max_norm=1)
    return framework