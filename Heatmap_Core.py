import torch
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FixedLocator
import seaborn
import wandb
import Utilities_Plotting

def createPlot():
    '''
        Parameters:
            None
        Returns :
            plot : the plot area for the graph
            axes : the axes lables of the graph
        Function:
            Creates a graph are to plot the heatmaps on
    '''
    plot,axes=plt.subplots(3,3,figsize=(15,15))
    plot.tight_layout(pad=5.0)
    plot.subplots_adjust(top=0.90)
    axes=axes.ravel()
    return plot,axes


def createAttentionPerCharacter(attention,character,bengaliLength,englishLength):
    '''
        Parameters:
            attention : attention matrix
            character : character of the attention matrix
            bengaliLength : length of target word
            englishLength : length of source word
        Returns :
            A tensor with the attention per character
        Function:
            Calculates attention per character
    '''
    att=attention[character]
    att=att[:bengaliLength]
    att=att[:,:englishLength]
    return att.T.cpu()


def createHeatMap(attentionPerCharacter,axes,row,bengaliLength,englishLength,xTicks,yTicks,fontName):
    '''
        Parameters:
            attentionPerCharacter : attention given to each character
            axes : axes of the graph to plot the heatmap
            row : which row of the grid of grpah to plot
            bengaliLength : length of the target word
            englishLength : length of the source word
            xTicks : x axis labels
            yTicks : y axis labels
            fontName : the font file name which will be used to support the translated language in matplotlib
        Returns :
            axes : axes of the graph after the heatmap is plotted on the particluar row
        Function:
            Creates heatmap for each position of the grid
    '''
    
    '''create the graph objects required'''
    nullObj=Utilities_Plotting.getNullObject()
    xObj=Utilities_Plotting.getFormatObject(xTicks)
    yObj=Utilities_Plotting.getFormatObject(yTicks)

    '''create the heatmap structure'''
    seaborn.heatmap(attentionPerCharacter,ax=axes[row],cmap='magma',cbar=False,vmin=0.0,vmax=1.0)

    '''edit the axes as per requirement'''
    axes[row].xaxis.set_major_formatter(nullObj)
    minorTickLocator=list()
    for pos in range(bengaliLength):
        minorTickLocator.append(pos+0.5)
    minorObj=FixedLocator(minorTickLocator)
    axes[row].xaxis.set_minor_locator(minorObj)
    axes[row].xaxis.set_minor_formatter(xObj)
    axes[row].yaxis.set_major_formatter(nullObj)
    minorTickLocator=list()
    for pos in range(englishLength):
        minorTickLocator.append(pos+0.5)
    minorObj=FixedLocator(minorTickLocator)
    axes[row].yaxis.set_minor_locator(minorObj)
    axes[row].yaxis.set_minor_formatter(yObj)
    axes[row].set_yticklabels(yTicks,rotation=0,fontdict={'fontsize':12})  
    axes[row].set_xticklabels(xTicks,fontproperties=FontProperties(fname=fontName),fontdict={'fontsize':12})
    axes[row].xaxis.tick_top()
    axes[row].set_xlabel('Predicted Bengali Word',size=14,labelpad=-300)
    axes[row].set_ylabel('Original English Word',size=14)
    return axes