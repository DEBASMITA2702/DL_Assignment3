import wandb
import warnings
warnings.filterwarnings("ignore")
import argparse
from LoadDataset_Test import DatasetLoad
from PrepareVocabulary_Final import PrepareVocabulary
from WordEmbeddings_Create import WordEmbeddings
from ModelTrainDriver_Framework import Model
import ModelForTestAttention_Run
import ModelForTest_Run
import Heatmap_Run
import Utilities_Tensor
import Utilities_Sequence
import train_arguments
import itertools
import torch.utils as utils
import random

'''login to wandb to generate plot'''
wandb.login()

'''main driver function'''
def main():
    '''default values of each of the hyperparameter. it is set according to the config of my best model'''
    project_name = 'Debasmita-DA6401-Assignment-3'
    entity_name = 'cs24m015-indian-institute-of-technology-madras' 
    modelType = "GRU"
    embeddingSize = 16
    neruonsInFC = 128
    layersInEncoder = 3
    layersInDecoder = 1
    bidirectional = "NO"
    learningRate = 0.001
    epochs = 10
    batchSize = 64
    dropoutProb = 0.2
    test = 1
    root = '/kaggle/input/dataset/Dakshina/bn/lexicons'
    attention = 1
    heatmap = 1
    fontName = '/kaggle/input/bengalifont/BengaliFont.ttf'

    '''call to argument function to get the arguments'''
    argumentsPassed = train_arguments.arguments()

    '''checking if a particular argument is passed through command line or not and updating the values accordingly'''
    if argumentsPassed.wandb_project is not None:
        project_name = argumentsPassed.wandb_project
    if argumentsPassed.wandb_entity is not None:
        entity_name = argumentsPassed.wandb_entity
    if argumentsPassed.cell is not None:
        modelType = argumentsPassed.cell
    if argumentsPassed.embedding is not None:
        embeddingSize = argumentsPassed.embedding
    if argumentsPassed.neurons is not None:
        neruonsInFC = argumentsPassed.neurons
    if argumentsPassed.encoder is not None:
        layersInEncoder = argumentsPassed.encoder
    if argumentsPassed.decoder is not None:
        layersInDecoder = argumentsPassed.decoder
    if argumentsPassed.bidir is not None:
        bidirectional = argumentsPassed.bidir
    if argumentsPassed.epochs is not None:
        epochs = argumentsPassed.epochs
    if argumentsPassed.batch is not None:
        batchSize = argumentsPassed.batch
    if argumentsPassed.dropout is not None:
        dropoutProb = argumentsPassed.dropout
    if argumentsPassed.test is not None:
        test = argumentsPassed.test
    if argumentsPassed.root is not None:
        root = argumentsPassed.root
    if argumentsPassed.attention is not None:
        attention = argumentsPassed.attention
    if argumentsPassed.heat is not None:
        heatmap = argumentsPassed.heat
    if argumentsPassed.font is not None:
        fontName = argumentsPassed.font

    '''initializing to the project'''
    wandb.init(project=project_name, entity=entity_name)

    '''calling the functions with the parameters'''
    if attention == 0:
        run = f"EP_{epochs}_CELL_{modelType}_EMB_{embeddingSize}_ENC_{layersInEncoder}_DEC_{layersInDecoder}_FC_{neruonsInFC}_DRP_{dropoutProb}_BS_{batchSize}_BIDIREC_{bidirectional}"
    else:
        run = f"ATT_YES_EP_{epochs}_CELL_{modelType}_EMB_{embeddingSize}_ENC_{layersInEncoder}_DEC_{layersInDecoder}_FC_{neruonsInFC}_DRP_{dropoutProb}_BS_{batchSize}_BIDIREC_{bidirectional}"
    print(f"run name = {run}")
    wandb.run.name = run

    Train.runTrain(root, epochs, batchSize, test, attention, heatmap, modelType, embeddingSize, layersInEncoder, layersInDecoder, neruonsInFC, bidirectional, dropoutProb, learningRate, fontName)
    wandb.finish()


class Train:
    def runTrain(
        root, epochs, batchSize, test, attention, heatmap,
        modelType, embeddingSize, layersInEncoder, layersInDecoder,
        neruonsInFC, bidirectional, dropoutProb, learningRate, fontName
    ):
        '''loads dataset'''
        lang = os.path.basename(os.path.dirname(root))
        d = DatasetLoad()
        d.loadDataset(root, lang)
        d.loadTestDataset(root, lang)
        
        '''creates vocabulary from the dataset'''
        vocabulary = PrepareVocabulary()
        vocabulary.createVocabulary(d.train_dataset)

        '''create embeddings of words for train, validation and test dataset'''
        embeddingTrain = WordEmbeddings()
        embeddingTrain.createWordEmbeddings(d.train_dataset, vocabulary)

        embeddingVal = WordEmbeddings()
        embeddingVal.createWordEmbeddings(d.val_dataset, vocabulary)

        embeddingTest = WordEmbeddings()
        embeddingTest.createWordEmbeddings(d.test_dataset, vocabulary)

        '''create the dataloaders'''
        trainEmbeddedDataset = utils.data.TensorDataset(
            embeddingTrain.englishEmbedding,
            embeddingTrain.bengaliEmbedding
        )
        trainEmbeddedDataLoader = utils.data.DataLoader(
            trainEmbeddedDataset, batch_size=64, shuffle=True
        )

        valEmbeddedDataset = utils.data.TensorDataset(
            embeddingVal.englishEmbedding,
            embeddingVal.bengaliEmbedding
        )
        valEmbeddedDataLoader = utils.data.DataLoader(
            valEmbeddedDataset, batch_size=64
        )

        testEmbeddedDataset = utils.data.TensorDataset(
            embeddingTest.englishEmbedding,
            embeddingTest.bengaliEmbedding
        )
        testEmbeddedDataset = utils.data.DataLoader(
            testEmbeddedDataset, batch_size=64
        )

        '''create an object of the encoder-decoder architecture with the best configuration for attention based model'''
        myModel = Model(
            vocabulary, trainEmbeddedDataLoader, valEmbeddedDataLoader,
            test=test, attention=attention, trainPy=1
        )
        framework = myModel.createModelFramework(
            modelType=modelType,
            embeddingSize=embeddingSize,
            neruonsInFC=neruonsInFC,
            layersInEncoder=layersInEncoder,
            layersInDecoder=layersInDecoder,
            dropout=dropoutProb,
            bidirectional=bidirectional,
            learningRate=learningRate,
            epochs=epochs,
            batchSize=batchSize
        )

        '''if prompted then do testing'''
        if test == 1:
            if attention == 1:
                '''call the function which calculates the accuracy and loss'''
                paramList = [
                    framework, testEmbeddedDataset, d.test_dataframe,
                    64, vocabulary.paddingIndex,
                    vocabulary.endOfSequenceIndex,
                    vocabulary.indexToCharDictForBengali
                ]
                ModelForTestAttention_Run.RunTestOnBestModel.testAndGivePredictions(
                    paramList, trainPy=1
                )

                '''if required then plot the heatmap'''
                if heatmap == 1:
                    englishLength = embeddingTest.englishEmbedding.size(1)
                    bengaliLength = embeddingTest.bengaliEmbedding.size(1)

                    requiredIndices = random.sample(range(4097), 9)

                    '''create zero tensros for stroing the words'''
                    englishWords = Utilities_Tensor.getLongZeroTensor(
                        len(requiredIndices), englishLength
                    )
                    bengaliWords = Utilities_Tensor.getLongZeroTensor(
                        len(requiredIndices), bengaliLength
                    )

                    '''store 9 source and their corresponding target words to create heatmap'''
                    for heatmapIndex, position in enumerate(requiredIndices):
                        batchPosition = Utilities_Sequence.getBatchFloorValue(position, 64)
                        datasetPosition = position - batchPosition * 64
                        data = next(
                            itertools.islice(testEmbeddedDataset, batchPosition, None)
                        )
                        englishWord, bengaliWord = data[0][datasetPosition], data[1][datasetPosition]
                        englishWords[heatmapIndex] = englishWord
                        bengaliWords[heatmapIndex] = bengaliWord
                        heatmapIndex += 1
                    Heatmap_Run.plotAttn(
                        framework, englishWords, bengaliWords,
                        vocabulary, trainPy=1, fontName=fontName
                    )
            else:
                '''call the function which calculates the accuracy and loss'''
                paramList = [
                    framework, testEmbeddedDataset, d.test_dataframe,
                    64, vocabulary.paddingIndex,
                    vocabulary.endOfSequenceIndex,
                    vocabulary.indexToCharDictForBengali
                ]
                ModelForTest_Run.RunTestOnBestModel.testAndGivePredictions(paramList, trainPy=1)



if __name__ == '__main__':
    main()
