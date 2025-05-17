import torch
import numpy as np

def increaseDimension(data):
    '''
        Parameters:
            data : tensor whose dimension to increase
        Returns :
            data : same tensor after dimension increase
        Function:
            Performs dimension increase in tensor
    '''
    return data.unsqueeze(0)


def decreaseDimension(data):
    '''
        Parameters:
            data : tensor whose dimension to decrease
        Returns :
            data : same tensor after dimension decrease
        Function:
            Performs dimension decrease in tensor
    '''
    return data.squeeze(0)


def expandTensor(tensor, dim1, dim2, dim3):
    '''
        Parameters:
            tensor : tensor whose dimensions are to be reproduced
            dim1,dim2,dim3 : dimensions along which to reproduce the tensor
        Returns :
            tensor : same tensor after reproducing dimension
        Function:
            Performs dimension reproducing in tensor
    '''
    return tensor.repeat(dim1, dim2, dim3)


def reorderDimensions(data, dim1, dim2, dim3):
    '''
        Parameters:
            data : tensor whose dimensions are to be reordered
            dim1,dim2,dim3 : dimensions along which to reorder the tensor
        Returns :
            data : same tensor after reordering dimension
        Function:
            Performs dimension reordering in tensor
    '''
    return data.permute(dim1, dim2, dim3)


def mutiplyTensors(tensor1, tensor2):
    '''
        Parameters:
            tensor1,tensor2 : the tensors which are to be multiplied
        Returns :
            a product of the two tensors
        Function:
            Performs tensor multiplication
    '''
    return tensor1 @ tensor2


def addTensor(tensor1, tensor2):
    '''
        Parameters:
            tensor1,tensor2 : the tensors which are to be added
        Returns :
            a sum of the two tensors
        Function:
            Performs tensor addition
    '''
    return tensor1 + tensor2


def concatenateTensor(tensor1, tensor2, dimension):
    '''
        Parameters:
            tensor1,tensor2 : the tensors which are to be concatenated
            dimension : dimension along which to concatenate
        Returns :
            a concatenated tensor
        Function:
            Performs tensor concatenation
    '''
    return torch.cat([tensor1, tensor2], dim=dimension)


def getMean(data):
    '''
        Parameters:
            data : tensor to find the mean
        Returns :
            mean of the tensor
        Function:
            Calculates the mean of tensor values
    '''
    return data.mean(axis=0)


def getShapeOfTensor(tensor, dimension):
    '''
        Parameters:
            tensor : tensor to find the shape
            dimension : which dimension to find the shape
        Returns :
            shape of the tensor along the dimension
        Function:
            Calculates the shape of tensor
    '''
    return tensor.shape[dimension]


def resizeTensor(tensor, dim1, dim2, dim3, orientation):
    '''
        Parameters:
            tensor : tensor to resize
            dim1,dim2,dim3 : dimensions along which to resize the tensor
            orientation : orientation of the tensor
        Returns :
            tensor : same tensor after resizing
        Function:
            Resizes a tensor
    '''
    return tensor.view(dim1, dim2, dim3, orientation)


def reverseTensor(tensor):
    '''
        Parameters:
            tensor : tensor to reverse
        Returns :
            same tensor after reversing
        Function:
            Reverses a tensor
    '''
    return tensor[-1]


def getZeroTensor(dim1, dim2, dim3):
    '''
        Parameters:
            dim1,dim2,dim3 : dimensions to form the tensor
        Returns :
            a zero tensor
        Function:
            Creates a zero tensor
    '''
    return torch.zeros(dim1, dim2, dim3)


def getLongZeroTensor(dim1, dim2):
    '''
        Parameters:
            dim1,dim2 : dimensions to form the tensor
        Returns :
            a long zero tensor
        Function:
            Creates a long zero tensor
    '''
    return torch.zeros(dim1, dim2, dtype=torch.long)


def extractColumn(tensor):
    '''
        Parameters:
            tensor : tensor to extract column
        Returns :
            same tensor after extracting column
        Function:
            Extracts column from tensor
    '''
    return tensor[1:]