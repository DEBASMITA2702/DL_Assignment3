import torch

def modification(modelEval, outputSequence):
    '''
        Parameters:
            modelEval : output from the model
            outputSequence : original target sequence
        Returns :
            modelEval : modified output to use it for other batches
            bengaliSequence : sequence in target language
        Function:
            Changes dimensions of the tensors
    '''
    dim = modelEval.shape[2]
    modelEvalSplit = modelEval[1:]
    modelEval = modelEvalSplit.reshape(-1, dim)
    bengaliSequenceSplit = outputSequence[1:]
    bengaliSequence = bengaliSequenceSplit.reshape(-1)
    return modelEval, bengaliSequence