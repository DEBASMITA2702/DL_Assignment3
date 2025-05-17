import pandas as pd
from Question5_c_Plot import main as plot_main

'''The purpose of this code is to act like a driver code for generating the plot of comparison between the attention model and the seq2seq model'''

def main():
    '''
        reading the two files
            modelPredictions.csv : contains all the predicted words by the vanilla seq2seq model
            modelPredictionsWithAttention.csv : contains all the predicted words by the attention model
    '''

    vanillaDataframe = pd.read_csv('modelPredictions.csv')                                                
    attentiondataFrame = pd.read_csv('modelPredictionsWithAttention.csv')

    '''setting the path where the file storing the comparison of the two models will be stored'''
    dataframeSavePath = "AttentionVsSeq2Seq.csv"

    '''
        creating a list to store the words.
        the words which are wrongly predicted by seq2seq model and correctly predicted by attention model are stored here
    '''
    container = list()
    '''iterating over the entire predictions'''
    for index, (row1, row2) in enumerate(zip(vanillaDataframe.iterrows(), attentiondataFrame.iterrows())):
        '''
            checking if seq2seq prediction is wrong and attention prediction is correct
            if yes then add the respective words into the list
        '''
        if row1[1]['Original'] != row1[1]['Predicted'] and row2[1]['Original'] == row2[1]['Predicted']:
            container.append((row1[1]['English'], row1[1]['Original'], row1[1]['Predicted'], row2[1]['Predicted']))

    '''creating a dataframe for the final csv file and putting the contents of the list created above into the dataframe'''
    finalDataframe = pd.DataFrame(container, columns=['English', 'Original', 'Seq2Seq', 'Attention'])

    '''saving the file into the path specified'''
    finalDataframe.to_csv(dataframeSavePath, index=False)

    plot_main()

if __name__ == "__main__":
    main()