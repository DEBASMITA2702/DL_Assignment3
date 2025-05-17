import Utilities_Plotting
import pandas as pd
import wandb

'''this code generates the plot of comparison between the attention model and the seq2seq model'''

def main():
    '''
        reading the file saved above and randomly picking 10 sample points to plot
    '''
    df = pd.read_csv('AttentionVsSeq2Seq.csv').sample(n=10)

    '''
        creating two lists to store the number of characters found different in the two models
        (our expectation is to get 0 differences for the attention words)
    '''
    differencesSeq2Seq = list()
    differencesAttention = list()

    '''iterating over the 10 sample points'''
    for _, row in df.iterrows():
        '''picking the original translation, seq2seq translation and the attention translation'''
        original = row['Original']
        seq2seq = row['Seq2Seq']
        attention = row['Attention']

        '''finding the number of difference by checking each character in the seq2seq translation and the original translation'''
        numberOfDifferences = 0
        for char1, char2 in zip(original, seq2seq):
            if char1 != char2:
                numberOfDifferences += 1
        differencesSeq2Seq.append(numberOfDifferences)

        '''finding the number of difference by checking each character in the attention translation and the original translation'''
        numberOfDifferences = 0
        for char1, char2 in zip(original, attention):
            if char1 != char2:
                numberOfDifferences += 1
        differencesAttention.append(numberOfDifferences)

    '''creating two columns in the dataframe for the respective differences of the two models'''
    df['Differences_Seq2Seq'] = differencesSeq2Seq
    df['Differences_Attention'] = differencesAttention

    '''calling the utility function to generate the image'''
    image = Utilities_Plotting.plotHtmlComparison(df, "AttentionVsSeq2Seq1.html")

    '''logging the plot into wandb'''
    wandb.login()
    wandb.init(project="Debasmita-DA6401-Assignment-3", name="Question 5 Attention Vs Seq2Seq")
    wandb.log({"Attention Vs Seq2Seq": wandb.Image(image)})
    wandb.finish()

if __name__ == "__main__":
    main()