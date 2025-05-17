import Utilities_Device_Trainings
import torch
import pandas as pd
from PIL import Image
from ModelForTestAttention_Utils import calculate, createCsv, createPlot

'''class to run the test on attention based model'''
class RunTestOnBestModel:
    def testAndGivePredictions(argList, trainPy=0):
        '''
            Parameters:
                argList : list of arguments
            Returns :
                image : image of the table generated
            Function:
                Runs test on the test dataset and gives accuracy and loss. Also stores the predicted words of the model in a csv.
                Also genertaes a table of 10 random data and show the number of mispredicted characters in each words (0 for true prediction)
        '''
        framework = argList[0]
        dataLoader = argList[1]
        actualData = argList[2]
        batchSize = argList[3]
        paddingIndex = argList[4]
        endOfSequenceIndex = argList[5]
        indexToCharDictForBengali = argList[6]

        modelPredictedWords = []
        framework.eval()

        '''set loss function'''
        lossFunction = Utilities_Device_Trainings.setLossFunction()

        totalLoss = 0.0
        correctPredictions = 0
        
        with torch.no_grad():
            '''iterate over the dataset'''
            for data in dataLoader:
                inputSequence = data[0]
                outputSequence = data[1]
                inputSequence = inputSequence.T
                inputSequence = Utilities_Device_Trainings.setDevice(inputSequence)
                outputSequence = outputSequence.T
                outputSequence = Utilities_Device_Trainings.setDevice(outputSequence)

                '''run the encoder-decoder architecture with no teacher forcing (as we are in inference step)'''
                modelEval, _ = framework(inputSequence, outputSequence, teacherRatio=0.0)

                '''calculate the correct predictions and loss for the current batch of data'''
                predictedSequence, correctBatch, lossBatch = calculate(modelEval, outputSequence, paddingIndex, lossFunction)
                correctPredictions += correctBatch
                totalLoss += lossBatch
                
                '''store the predictions of the model'''
                predictedSequence = predictedSequence.T
                for pos in range(batchSize):
                    word = ""
                    try:
                        for predictedChar in predictedSequence[pos]:
                            if predictedChar >= paddingIndex:
                                charIndex = predictedChar.item()
                                if charIndex in indexToCharDictForBengali:
                                    word += indexToCharDictForBengali[charIndex]
                            if predictedChar == endOfSequenceIndex:
                                break
                    except IndexError as e:
                        continue 
                    modelPredictedWords.append(word)

            '''calculate accuracy'''
            testAccuracy = correctPredictions / (len(dataLoader) * batchSize)

            if trainPy == 0:
                print("Test Accuracy for best model with attention: {}".format(testAccuracy))
            else:
                print("Test Accuracy with attention: {}".format(testAccuracy))

            '''create csv of the predictions'''
            createCsv(actualData, modelPredictedWords)                

            if trainPy == 0:
                '''create the image of the table'''
                createPlot()                                  
        
                image = Image.open("predictions_attention/ModelPredictionsAttention.png")
                return image