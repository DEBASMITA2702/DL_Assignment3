import torch
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FixedLocator
import seaborn
import wandb
import Utilities_Device_Trainings
import Utilities_Plotting
import Utilities_Tensor
from Heatmap_Core import createPlot, createAttentionPerCharacter, createHeatMap

def plotAttn(model,inputSequence,outputSequence,vocabulary,trainPy=0,fontName='/kaggle/input/bengalifont/BengaliFont.ttf'):    
    '''
        Parameters:
            model : model on which to create the heatmaps
            inputSequence : word in source language
            outputSequence : word in target language
            vocabulary : vocabulary of the dataset
            trainPy : variable indicating whether this is train.py call or not
            fontName : the font file name which will be used to support the translated language in matplotlib
        Returns :
            None
        Function:
            Creates 3x3 heatmap grid
    '''
    model.eval()    
    with torch.no_grad():
        '''get the original source and target words'''
        inputSequence=inputSequence.T
        inputSequence=Utilities_Device_Trainings.setDevice(inputSequence)
        outputSequence=outputSequence.T
        outputSequence=Utilities_Device_Trainings.setDevice(outputSequence)

        '''run the encoder-decoder architecture and get the model predictions'''
        modelEval,attention=model(inputSequence,outputSequence,teacherRatio=0.0)
        
        modelEval=Utilities_Tensor.extractColumn(modelEval)
        attention=Utilities_Tensor.extractColumn(attention)
        attentionSequence=modelEval.argmax(dim=2)
        
        attention=Utilities_Tensor.reorderDimensions(attention,1,0,2)
        inputSequence=inputSequence.T
        attentionSequence=attentionSequence.T

        _,axes=createPlot()
        
        '''iterate on each character of the word'''
        for row in range(inputSequence.size(0)):
            englishLength=inputSequence.size(1)
            bengaliLength=outputSequence.size(1)-1
            
            '''source word'''
            column=0
            flag=True
            while(flag and column<inputSequence.size(1)):
                if inputSequence[row][column].item()==vocabulary.endOfSequenceIndex:
                    englishLength=column+1
                    flag=False
                column+=1
            
            '''target word'''
            column=0
            flag=True
            while(flag and column<outputSequence.size(1)-1):
                if attentionSequence[row][column].item()==vocabulary.endOfSequenceIndex:
                    bengaliLength=column+1
                    flag=False
                column+=1

            '''calculate attention per character'''
            attentionPerCharacter=createAttentionPerCharacter(attention,row,bengaliLength,englishLength)

            '''generate the x and y labels'''
            xTicks,yTicks=Utilities_Plotting.createXandYticks(bengaliLength,englishLength,vocabulary,attentionSequence,inputSequence,row)

            '''create the heatmap'''
            if row<len(axes):
                axes=createHeatMap(attentionPerCharacter,axes,row,bengaliLength,englishLength,xTicks,yTicks,fontName)

        '''save the plot and log into wandb'''
        if trainPy==0:
            plt.savefig('AttentionHeatMap.png')
        else:
            plt.savefig('AttentionHeatMap1.png')
        
        if trainPy==0:
            wandb.login()
            wandb.init(project="Debasmita-DA6401-Assignment-3",name="Question 5 Attention Heatmap")
        
        wandb.log({'Attention Heatmap':wandb.Image(plt)})
        
        if trainPy==0:
            wandb.finish()
        plt.close()