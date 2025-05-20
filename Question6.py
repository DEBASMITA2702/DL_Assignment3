import pandas as pd
import numpy as np
import torch.utils as utils
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import wandb
from matplotlib.colors import Normalize

from LoadDataset_Test import DatasetLoad
from PrepareVocabulary_Final import PrepareVocabulary
from WordEmbeddings_Create import WordEmbeddings
from ModelTrainDriver_Framework import Model
from AttentionWeightsFetch import plotAttn  # utility to extract attention weights

def main():
    '''load dataset'''
    lang = "bn"
    d = DatasetLoad()
    root = "/kaggle/input/dataset/Dakshina/bn/lexicons"
    d.loadDataset(root, lang)
    d.loadTestDataset(root, lang)

    '''build the vocabulary'''
    vocabulary = PrepareVocabulary()
    vocabulary.createVocabulary(d.train_dataset)

    '''create embeddings for train, val, test'''
    embeddingTrain = WordEmbeddings()
    embeddingTrain.createWordEmbeddings(d.train_dataset, vocabulary)

    embeddingVal = WordEmbeddings()
    embeddingVal.createWordEmbeddings(d.val_dataset, vocabulary)

    embeddingTest = WordEmbeddings()
    embeddingTest.createWordEmbeddings(d.test_dataset, vocabulary)

    '''wrapoing the embeddings in DataLoaders'''
    train_dataset = utils.data.TensorDataset(
        embeddingTrain.englishEmbedding,
        embeddingTrain.bengaliEmbedding
    )
    train_loader = utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    val_dataset = utils.data.TensorDataset(
        embeddingVal.englishEmbedding,
        embeddingVal.bengaliEmbedding
    )
    val_loader = utils.data.DataLoader(
        val_dataset,
        batch_size=64
    )

    '''instantiating the best attention model'''
    model_att = Model(
        vocabulary,
        train_loader,
        val_loader,
        test=1,
        attention=1
    )
    model_att.createModelFramework(
        modelType="GRU",
        embeddingSize=16,
        neruonsInFC=128,
        layersInEncoder=3,
        layersInDecoder=1,
        dropout=0.2,
        bidirectional="NO",
        learningRate=0.001,
        epochs=10,
        batchSize=64
    )
    model_att.framework.eval()

    csv_path = "/kaggle/input/seqtoseqcsv/AttentionVsSeq2Seq.csv"
    df = pd.read_csv(csv_path)
    correct_df = df[df['Original'] == df['Attention']].copy()
    if correct_df.empty:
        raise RuntimeError("No exact matches in the CSV file")
    correct_df['eng_len'] = correct_df['English'].str.len()

    '''pick one random correct prediction'''
    row = correct_df.sample(n=1).iloc[0]
    eng_word = row['English']
    beng_word = row['Attention']    
    print(f"\nPicked the word → English: '{eng_word}' | Predicted Bengali: '{beng_word}'\n")

    '''extract the true attention matrix'''
    attn_matrix = plotAttn(
        model_att.framework,
        [eng_word],
        [beng_word],
        vocabulary
    )    

    '''creating the heatmap'''
    font_path = '/kaggle/input/bengalifont/BengaliFont.ttf'
    prop = fm.FontProperties(fname=font_path)

    baseline = 0.01 * np.max(attn_matrix)
    enhanced = np.clip(attn_matrix + baseline, 0.0, 1.0)

    plt.figure(figsize=(8, 6))
    plt.imshow(enhanced,aspect='auto',cmap='YlGnBu',norm=Normalize(vmin=0.0, vmax=1.0))
    cbar = plt.colorbar()
    cbar.set_label('', fontsize=12)

    plt.xticks(np.arange(len(eng_word)), list(eng_word), fontsize=14)
    plt.yticks(np.arange(len(beng_word)), list(beng_word),fontproperties=prop, fontsize=16)
    plt.xlabel(f"English: {eng_word}", fontsize=16)
    plt.ylabel(f"বাংলা: {beng_word}", fontproperties=prop, fontsize=16)
    plt.title("Attention Connectivity", fontsize=18)

    for i in range(enhanced.shape[0]):
        for j in range(enhanced.shape[1]):
            val = enhanced[i, j]
            color = 'white' if val < 0.5 else 'black'
            plt.text(j, i, f"{val:.2f}",
                     ha='center', va='center',
                     color=color, fontproperties=prop, fontsize=10)

    plt.grid(which='minor', linestyle='-', linewidth=0.5, color='gray')
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()

    '''Wandb logging'''
    wandb.login()
    run = wandb.init(
        project="Debasmita-DA6401-Assignment-3", 
        name="Question6_Visualisation"
    )
    run.log({"Attention_Matrix": wandb.Image(fig)})
    run.finish()

if __name__ == '__main__':
    main()