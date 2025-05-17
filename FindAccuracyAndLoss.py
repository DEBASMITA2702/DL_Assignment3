import Utilities_Device_Trainings
import torch
from AccuracyAndLoss_Utils import calculate

'''class to find the accuracy and loss'''
class FindAccuracyAndLoss:
    def findAccuracyAndLoss(framework, dataLoader, batchSize, paddingIndex):
        '''
            Parameters:
                framework : object of the architecture
                dataLoader : data on which to calculate the accuracy and loss
                batchSize : batch size used
                paddingIndex : encoding of the padding characters in the vocabulary
            Returns :
                averageLoss : average loss across the dataset
                accuracy : accuracy of correct prediction 
            Function:
                Calculates the accuracy percentage and average loss for the dataset
        '''

        '''sets loss function'''
        framework.eval()
        lossFunction = Utilities_Device_Trainings.setLossFunction()
    
        totalLoss = 0.0
        correctPredictions = 0
        
        with torch.no_grad():
            '''iterate the whole dataset'''
            for _, data in enumerate(dataLoader):
                '''get the original source and target word'''
                inputSequence = data[0]
                outputSequence = data[1]
                inputSequence = inputSequence.T
                inputSequence = Utilities_Device_Trainings.setDevice(inputSequence)
                outputSequence = outputSequence.T
                outputSequence = Utilities_Device_Trainings.setDevice(outputSequence)

                '''run the encoder-decoder architecture with no teacher forcing (as we are in inference step)'''
                modelEval, _ = framework(inputSequence, outputSequence, teacherRatio=0.0)
                
                '''calculate the correct predictions and loss for the current batch of data'''
                modelEval, correctBatch, lossBatch = calculate(modelEval, outputSequence, paddingIndex, lossFunction)
                correctPredictions += correctBatch
                totalLoss += lossBatch
            
            '''avergae loss and accuracy percentage'''
            accuracy = correctPredictions / (len(dataLoader) * batchSize)
            averageLoss = totalLoss / len(dataLoader)
            return averageLoss, accuracy