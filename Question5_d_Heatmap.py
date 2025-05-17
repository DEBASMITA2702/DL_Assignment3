from ModelTrainDriver_Framework import Model
import Heatmap_Run

def plot_heatmaps(vocabulary, trainEmbeddedDataLoader, valEmbeddedDataLoader, englishWords, bengaliWords):
    '''create an object of the model and run the architecture with the best configuration'''    
    modelBestWithAttention=Model(vocabulary,trainEmbeddedDataLoader,valEmbeddedDataLoader,test=1,attention=1)
    modelBestWithAttention.createModelFramework(modelType="GRU",embeddingSize=16,neruonsInFC=128,
                                                layersInEncoder=3,layersInDecoder=1,dropout=0.2,bidirectional="NO",
                                                learningRate=0.001,epochs=10,batchSize=64)

    '''plot the heatmaps'''
    Heatmap_Run.plotAttn(modelBestWithAttention.framework,englishWords,bengaliWords,vocabulary)