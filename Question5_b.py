from LoadDataset_Test import DatasetLoad
from PrepareVocabulary_Final import PrepareVocabulary
from WordEmbeddings_Create import WordEmbeddings
from ModelTrainDriver_Framework import Model
from ModelForTestAttention_Run import RunTestOnBestModel
import torch.utils as utils
import wandb

'''purpose of this code is to test the best attention based model'''
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

    '''create an object of the encoder-decoder architecture with the best configuration for attention based model'''
    modelBestWithAttention=Model(vocabulary,trainEmbeddedDataLoader,valEmbeddedDataLoader,test=1,attention=1)    
    modelBestWithAttention.createModelFramework(modelType="GRU",embeddingSize=16,neruonsInFC=128,
                                                layersInEncoder=3,layersInDecoder=1,dropout=0.2,bidirectional="NO",
                                                learningRate=0.001,epochs=10,batchSize=64)
    
    '''call the function which calculates the accuracy and loss'''
    paramList=[modelBestWithAttention.framework,testEmbeddedDataset,d.test_dataframe,64,vocabulary.paddingIndex,vocabulary.endOfSequenceIndex,vocabulary.indexToCharDictForBengali]
    
    
    image=RunTestOnBestModel.testAndGivePredictions(paramList)
    
    '''plot the image to wandb'''
    wandb.login()
    wandb.init(project="Debasmita-DA6401-Assignment-3",name="Question 5 Attention Predictions")
    wandb.log({"Attention Predictions":wandb.Image(image)})
    wandb.finish()


if __name__ == "__main__":
    main()