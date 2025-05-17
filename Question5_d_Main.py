from LoadDataset_Test import DatasetLoad
from PrepareVocabulary_Final import PrepareVocabulary
from WordEmbeddings_Create import WordEmbeddings
import Utilities_Tensor
import Utilities_Sequence
import torch.utils as utils
import itertools
import pandas as pd
from Question5_d_Heatmap import plot_heatmaps

def main():
    '''loads dataset'''
    lang="bn"
    d=DatasetLoad()
    root="/kaggle/input/dataset/Dakshina/bn/lexicons"
    d.loadDataset(root,lang)
    d.loadTestDataset(root,lang)

    '''creates vocabulary from the dataset'''
    vocabulary=PrepareVocabulary()
    vocabulary.createVocabulary(d.train_dataset)

    '''create embeddings of words for train, validation and test dataset'''
    embeddingTrain=WordEmbeddings()
    embeddingTrain.createWordEmbeddings(d.train_dataset,vocabulary)

    embeddingVal=WordEmbeddings()
    embeddingVal.createWordEmbeddings(d.val_dataset,vocabulary)

    embeddingTest=WordEmbeddings()
    embeddingTest.createWordEmbeddings(d.test_dataset,vocabulary)

    '''create the dataloaders'''
    trainEmbeddedDataset=utils.data.TensorDataset(embeddingTrain.englishEmbedding,embeddingTrain.bengaliEmbedding)
    trainEmbeddedDataLoader=utils.data.DataLoader(trainEmbeddedDataset,batch_size=64,shuffle=True)

    valEmbeddedDataset=utils.data.TensorDataset(embeddingVal.englishEmbedding,embeddingVal.bengaliEmbedding)
    valEmbeddedDataLoader=utils.data.DataLoader(valEmbeddedDataset,batch_size=64)

    testEmbeddedDataset=utils.data.TensorDataset(embeddingTest.englishEmbedding,embeddingTest.bengaliEmbedding)
    testEmbeddedDataset=utils.data.DataLoader(testEmbeddedDataset,batch_size=64)

    '''setting the length of the source (english in this case) and the target (bengali in this case) embeddings'''
    englishLength=embeddingTest.englishEmbedding.size(1)
    bengaliLength=embeddingTest.bengaliEmbedding.size(1)

    '''read the files and create dataframe'''
    dataFrame=pd.read_csv("AttentionVsSeq2Seq.csv")                          
    englishWordSelected=dataFrame.sample(n=9,random_state=42).iloc[:,0]
    dataFrame2=pd.read_csv("modelPredictionsWithAttention.csv")                       
    
    requiredIndices=dataFrame2[dataFrame2['English'].isin(englishWordSelected)]
    requiredIndices=requiredIndices.index.tolist()

    '''create zero tensros for stroing the words'''
    englishWords=Utilities_Tensor.getLongZeroTensor(len(requiredIndices),englishLength)
    bengaliWords=Utilities_Tensor.getLongZeroTensor(len(requiredIndices),bengaliLength)

    '''store 9 source and their corresponding target words to create heatmap'''
    for heatmapIndex,position in enumerate(requiredIndices):
        batchPosition=Utilities_Sequence.getBatchFloorValue(position,64)
        datasetPosition=position-batchPosition*64
        data=next(itertools.islice(testEmbeddedDataset,batchPosition,None))
        englishWord,bengaliWord=data[0][datasetPosition],data[1][datasetPosition]
        englishWords[heatmapIndex]=englishWord
        bengaliWords[heatmapIndex]=bengaliWord
        heatmapIndex+=1

    plot_heatmaps(vocabulary, trainEmbeddedDataLoader, valEmbeddedDataLoader, englishWords, bengaliWords)

if __name__ == "__main__":
    main()