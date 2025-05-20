# DA6401_Assignment3

## Getting the code files
You need to first clone the github repository containing the files.
```
git clone https://github.com/DEBASMITA2702/DL_Assignment3.git
```
Then change into the code directory.
```
cd DL_Assignment3
```
Make sure you are in the correct directory before proceeding further.


## Setting up the platform and environment
- ### Local machine
  If you are running the code on a local machine, then you need to have python installed in the machine and pip command added in the environemnt variables.
  You can execute the following command to setup the environment and install all the required packages
  ```
  pip install -r requirements.txt
  ```
- ### Google colab/Kaggle
  If you are using google colab platform or kaggle, then you need to execute the follwoing code
  ```
  pip install wandb argparse torch plotly matplotlib numpy pandas seaborn
  ```
This step will setup the environment required before proceeding.


## Project
The project deals in working with Sequence Learning Problems. It works with three types of cells:
- Recurrent Neural Networks (RNN)
- Long Short Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

The project supports all the three types of cells with the encoder and decoder architectures. Additionally the feature of adding attention to the encoder outputs before passing them to the decoder is also supported.

## Loading the dataset
- Dataset is placed inside the project ```DL_Assignment3``` directory
  - Target language is same as default language (Bengali)

    You do not need to do anything specifically for the dataset. The code will handle it automatically. You can simply run
    ```
    python train.py <any_specifications_related_to_the_model>
    ```
  - Target language is different

    You need to specify the path of the specific language directory inside the dataset. For example if you want to run on Telugu dataset then run it like :
    ```
    python train.py --root Dakshina/tel/lexicons
    ```

- Dataset is placed outside the project directory

  In this case you need to specify the absolute path of the directory of the language (even if it is the default language Bengali) you want to run the model with.
  For example,
  ```
  python train.py --root <absolute_path_of_the_specific_language_directory_inside_Dakshina_dataset>
  ```

#### Note
Here Dakshina refers to the directory obtained after unzipping the ```Dakshina.zip``` provided along with the question.


## Training the model

To train the model, you need to compile and execute the [train.py](https://github.com/DEBASMITA2702/DL_Assignment3/blob/main/train.py) file, and pass additional arguments if and when necessary.\
It can be done by using the command:
```
python train.py
```
By the above command, the model will run with the default configuration.\
To customize the run, you need to specify the parameters like ```python train.py <*args>```\
For example,
```
python train.py -e 20 -b 128 --cell GRU
```

The arguments supported are :
|           Name           | Default Value | Description                                                               |
| :----------------------: | :-----------: | :------------------------------------------------------------------------ |
| `-wp`, `--wandb_project` | Debasmita-DA6401-Assignment-3 | Project name used to track experiments in Weights & Biases dashboard      |
|  `-we`, `--wandb_entity` |     cs24m015-indian-institute-of-technology-madras    | Wandb Entity used to track experiments in the Weights & Biases dashboard |
|     `-r`,`--root`        |Dakshina/bn/lexicons |Absolute path of the specific language in the dataset                                         |
|     `-e`, `--epochs`     |       10      | Number of epochs to train neural network                                 |
|   `-b`, `--batch`        |       64       | Batch size to divide the dataset                                  |
|   `-n`, `--neurons`        |       128       | Number of neurons in the fully connected layer                                  |
|   `-d`, `--dropout`        |       0.2       | Percentage of dropout in the network                                  |
|   `-em`, `--embedding`        |       16       | Size of the embedding layer                                  |
|   `-enc`, `--encoder`        |       3       | Number of layers in the encoder                                  |
|   `-dec`, `--decoder`        |       1       | Number of layers in the decoder                                  |
|   `-c`, `--cell`        |       GRU       | Type of cell                                  |
|   `-bid`, `--bidir`        |       NO       | choices: [YES,NO]                                  |
|   `-t`, `--test`        |       1       | choices: [0,1]                                  |
|   `-att`, `--attention`        |       1       | choices: [0,1]                                  |
|   `-ht`, `--heat`        |       1       | choices: [0,1]                                  |
|   `-f`, `--font`        |       BengaliFont.TTF       | Font of the language chosen to generate the heatmap                                  |

The arguments can be changed as per requirement through the command line.
  - If prompted to enter the wandb login key, enter the key in the interactive command prompt.

## Testing the model
To test the model, you need to specify the test argument as 1. For example
```
python train.py -t 1
```
This will run the model with default parameters and print the test accuracy and loss.


## Code Organization
- WordEmbeddings_Create.py: Subclasses the base implementation to provide createWordEmbeddings(), which builds and pads embedding tensors for both English and Bengali datasets.
- WordEmbeddings_Translate.py: Defines translateWordToTensor(), converting individual words into character‐index tensors (with start/end tokens) based on the provided vocabulary.

- EncoderArchitecture.py: Defines the EncoderStack class’s constructor (__init__), setting up all model parameters, dropout, RNN cell, and embedding layer.
- EncoderArchitecture_Forward.py: Imports that base class and adds the forward() method implementation, handling embedding lookup, RNN evaluation, bidirectional processing, and attention.

- DecoderArchitecture.py: Sets up the decoder’s constructor (__init__), building RNN cell, dropout, embedding and FC layers.
- DecoderArchitecture_Forward.py subclasses it to provide the forward() logic with attention, gating, and output generation.

- CombinedModelArchitecture_Utils.py: Contains the helper functions formMatrix (zero-tensor creation) and doTeacherForcing.
- CombinedModelArchitecture_stack.py: Defines the EncoderDecoderStack class, importing and using those utilities to implement the encoder–decoder forward pass.

- AccuracyAndLoss_Utils.py: Contains the calculate function that computes batch-level loss and correct prediction count.
- FindAccuracyAndLoss.py: Defines the FindAccuracyAndLoss class with findAccuracyAndLoss(), which uses calculate to evaluate an entire dataset.

- RunTrainer_Utils.py: Contains the modification() helper function for reshaping model outputs and target sequences.
- RunTrainer.py: Defines the Trainer class with runModelTrainer(), driving the full training loop and logging.

- LoadDataset_Train.py: Defines loadDataset(), which loads training and validation CSVs into self.train_dataset and self.val_dataset.
- LoadDataset_Test.py: Defines loadTestDataset(), declares the DatasetLoad class, and attaches both loadDataset (imported) and loadTestDataset methods to it.

- PrepareVocabulary_Final.py: Defines PrepareVocabulary with its constructor (__init__), sets up token indices and attaches the split methods.
- PrepareVocabulary_Initialize.py: Implements initializeVocabularyDictionaries(self), initializing start/end/padding tokens in the English and Bengali dictionaries.
- PrepareVocabulary_Create.py: Implements createVocabulary(self, dataset), iterating over the dataset to build and count character vocabularies.

- ModelTrainDriver_Initialize.py: Defines the Model class with its constructor (__init__), setting up dataset handles, vocabulary sizes, and run flags.
- ModelTrainDriver_Framework.py: Imports Model, defines and attaches the createModelFramework() method, implementing encoder/decoder creation, training, and testing logic.

- ModelForTest_Utils.py: Exports calculate(), createCsv(), and createPlot() utility functions for computing metrics and generating CSV/plots.
- ModelForTest_Run.py: Defines the RunTestOnBestModel class with testAndGivePredictions(), which imports and uses those utilities to run inference, evaluate, and output results.

- ModelForTestAttention_Utils.py: Contains the standalone functions calculate(), createCsv(), and createPlot() for evaluating and visualizing attention-based model predictions.
- ModelForTestAttention_Run.py: Defines the RunTestOnBestModel class with testAndGivePredictions(), which imports and uses those utility functions to run tests and output results.

- Question2.py: Sets up and runs a Weights & Biases hyperparameter sweep for the vanilla encoder–decoder model by loading datasets, building vocabulary and embeddings, creating DataLoaders, naming runs, and launching a Bayes sweep agent for training.

- Question3_4.py: Tests the best-configured vanilla encoder–decoder model by loading datasets, preparing vocabulary and embeddings, constructing DataLoaders, running inference via RunTestOnBestModel, and logging prediction images to Wandb.

- Question5_a.py: Sets up and runs a hyperparameter sweep for the attention-enabled encoder–decoder model by loading datasets, preparing vocabulary and embeddings, creating DataLoaders, naming runs, and initiating the sweep agent.

- Question5_b.py: Runs inference on the best attention-based encoder–decoder model by loading datasets, preparing vocabulary and embeddings, building DataLoaders, instantiating the model with optimal settings, generating predictions via RunTestOnBestModel, and logging the results to Wandb.

- Question5_c_Main.py: Loads the vanilla and attention prediction CSVs, identifies which cases attention corrected seq2seq’s errors, and saves a combined CSV.
- Question5_c_Plot.py: Reads that combined CSV, samples 10 entries, computes per-word character-differences for both models, generates an HTML comparison plot, and logs it to Weights & Biases.

- Heatmap_Core.py: Provides low‐level routines (createPlot, createAttentionPerCharacter, createHeatMap) for constructing subplots, slicing attention matrices, and drawing individual heatmap panels.
- Heatmap_Run.py: Implements plotAttn(), which wires together your model’s forward pass, invokes the core plotting functions for each sample, saves the figure, and logs it to Weights & Biases.

- Question5_d_Main.py: Handles all data loading, vocabulary and embedding setup, and selects the specific word indices to visualize.
- Question5_d_Heatmap.py: Takes the prepared data, instantiates the best‐configuration model, runs it, and finally plots the attention heatmaps.

- AttentionWeightsFetch.py: Provides a utility function to extract, process, and plot character-level attention weight matrices from a trained encoder–decoder model—handling tensor reshaping, index-to-character mapping, and rendering heatmaps with matplotlib and seaborn.
- Question6.py: Selects a random correctly translated word pair from my generated .csv, invokes the best attention-enabled model to retrieve its attention matrix, visualizes the attention heatmap over English–Bengali characters, and logs the figure to Wandb.

- train_argument.py: Contains the full arguments() function for parsing CLI options.
- train.py: Imports arguments from train_args.py, and acts as the main training driver, parsing command-line hyperparameter overrides, initializing a W&B run, loading and embedding datasets, building and training encoder–decoder models (with optional attention and heatmap steps), and executing testing routines including attention-based predictions and visualizations.

-- I’ve divided Utilities.py into five logical modules---------
- Utilities_Device_Trainings.py: Environment & training helpers (setDevice, setOptimizer, setLossFunction, setOutputFunction, clipGradient).
- Utilities_Layer.py: Layer factories (createEmbeddingLayer, createLinearLayer, createDropoutLayer).
- Utilities_Tensor.py: Core tensor operations (dimension, replication, reshaping, arithmetic, extraction).
- Utilities_Sequence.py: Sequence and decoder helpers (runDecoderWithNoTeacherForcing, getBatchFloorValue).
- Utilities_Plotting.py: HTML table plotting and formatter utilities (plotHtml, plotHtmlComparison, createXandYticks, getNullObject, getFormatObject).


## Additional features
The following features are also supported
  - If you need some clarification on the arguments to be passed, then you can do
    ```
    python train.py --help
    ```
  - If you want to apply attention and generate the 3x3 heatmap grid then you can run the model like below. This will save a png file in the same working directory and also log the image into the wandb project
    ```
    python train.py --attention 1 --heatmap 1
    ``` 
  - If you are using any language other than the default Bengali language set for the model, then to generate the heatmap you need to download and install the font of the language into your environment. You can find the required font in [Font](https://fonts.google.com). Here you can type in your preferred language and download the respective font file. Right click on the downloaded file and install it. Make sure you download the file into the same directory as the project. After successful installation, you need to specify the name of the .TTF file generated and run the model like :
    ```
    python train.py --font <name_of_the_font_file>.TTF
    ```
  

## Links
[Wandb Report](https://wandb.ai/cs24m015-indian-institute-of-technology-madras/Debasmita-DA6401-Assignment-3/reports/DA6401-Assignment-3-Debasmita-Mandal-CS24M015--VmlldzoxMjQwMTM5Ng?accessToken=0o0ypuluohtuyr8hsjayzqfsr01inaxfjj80o16q6idz99x105g5ux86w28xfzuf)
