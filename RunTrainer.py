import torch
from FindAccuracyAndLoss import FindAccuracyAndLoss
import Utilities_Device_Trainings
from torch.nn.utils import clip_grad_norm_
import wandb
from copy import deepcopy
from RunTrainer_Utils import modification

'''class to run the epochs on the model'''
class Trainer:

    def runModelTrainer(paramList, trainPy=0, saveBestModel=0, logging=1):
        '''
            Parameters:
                paramList : list of parameters passes
                trainPy : variable indicating whether to this is train.py call or not
                saveBestModel : variable indicating whether to save the model or not
            Returns :
                None
            Function:
                Drives the training process and run epochs
        '''

        '''set the parameters'''
        framework = paramList[0]
        learningRate = paramList[1]
        trainEmbeddedDataLoader = paramList[2]
        valEmbeddedDataLoader = paramList[3]
        epochs = paramList[4]
        batchSize = paramList[5]
        paddingIndex = paramList[6]

        '''declare lists for storing the accuracies and losses'''
        trainAccuracyPerEpoch = list()
        trainLossPerEpoch = list()
        valAccuracyPerEpoch = list()
        valLossPerEpoch = list()

        '''setting the optimizer'''
        backpropagationFramework = Utilities_Device_Trainings.setOptimizer(framework, learningRate)

        '''setting the loss function'''
        lossFunction = Utilities_Device_Trainings.setLossFunction()

        '''run epochs'''
        for epoch in range(epochs):
            framework.train()
            for id, data in enumerate(trainEmbeddedDataLoader):
                '''get the original source and target words'''
                inputSequence = data[0]
                outputSequence = data[1]
                inputSequence = inputSequence.T
                inputSequence = Utilities_Device_Trainings.setDevice(inputSequence)
                outputSequence = outputSequence.T
                outputSequence = Utilities_Device_Trainings.setDevice(outputSequence)

                '''run the encoder-decoder architecture'''
                modelEval, _ = framework(inputSequence, outputSequence)
                modelEval, bengaliSequence = modification(modelEval, outputSequence)

                '''run backpropagation'''
                backpropagationFramework.zero_grad()
                loss = lossFunction(modelEval, bengaliSequence)
                loss.backward()
                clip_grad_norm_(framework.parameters(), max_norm=1)
                backpropagationFramework.step()
            
            '''calculate the respective loss and accuracy'''
            trainingLoss, trainingAccuracy = FindAccuracyAndLoss.findAccuracyAndLoss(
                framework, trainEmbeddedDataLoader, batchSize, paddingIndex
            )
            valLoss, valAccuracy = FindAccuracyAndLoss.findAccuracyAndLoss(
                framework, valEmbeddedDataLoader, batchSize, paddingIndex
            )
            
            trainLossPerEpoch.append(trainingLoss)
            trainAccuracyPerEpoch.append(trainingAccuracy)
            valLossPerEpoch.append(valLoss)
            valAccuracyPerEpoch.append(valAccuracy)

            if logging == 1:
                '''print and log the losses and accuracies to terminal and wandb respectively'''
                print("\n===================================================================================================================")
                print(f"Epoch : {epoch+1}")
                print(f"Training Accuracy : {trainAccuracyPerEpoch[-1]}")
                print(f"Validation Accuracy : {valAccuracyPerEpoch[-1]}")
                print(f"Training Loss : {trainLossPerEpoch[-1]}")
                print(f"Validation Loss : {valLossPerEpoch[-1]}")  
                wandb.log({
                    "training_accuracy": trainAccuracyPerEpoch[-1],
                    "validation_accuracy": valAccuracyPerEpoch[-1],
                    "training_loss": trainLossPerEpoch[-1],
                    "validation_loss": valLossPerEpoch[-1],
                    "Epoch": epoch+1
                })
            else:
                if epoch == epochs - 1:
                    return framework

        '''save the model if needed'''
        if saveBestModel == 1:
            state = deepcopy(framework.state_dict())
            torch.save(state, "/frameworkState.pth")
        
        if trainPy == 1:
            return framework