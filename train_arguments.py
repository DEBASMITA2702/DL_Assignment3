import argparse

'''
  Parameters:
    None
  Returns :
    A parser object
  Function:
    Does command line argument parsing and returns the arguments passed
'''
def arguments():
    commandLineArgument = argparse.ArgumentParser(description='Model Parameters')
    commandLineArgument.add_argument('-wp','--wandb_project',help="Project name used to track experiments in Weights & Biases dashboard")
    commandLineArgument.add_argument('-we','--wandb_entity',help="Wandb Entity used to track experiments in the Weights & Biases dashboard")
    commandLineArgument.add_argument('-r','--root',help="Absolute path of the dataset")
    commandLineArgument.add_argument('-e','--epochs',type=int,help="Number of epochs to train neural network")
    commandLineArgument.add_argument('-b','--batch',type=int,help="Batch size to divide the dataset")
    commandLineArgument.add_argument('-n','--neurons',type=int,help="Number of neurons in the fully connected layer")
    commandLineArgument.add_argument('-d','--dropout',type=float,help="Percentage of dropout in the network")
    commandLineArgument.add_argument('-em','--embedding',type=int,help="Size of the embedding layer")
    commandLineArgument.add_argument('-enc','--encoder',type=int,help="Number of layers in the encoder")
    commandLineArgument.add_argument('-dec','--decoder',type=int,help="Number of layers in the decoder")
    commandLineArgument.add_argument('-c','--cell',help="Type of cell")
    commandLineArgument.add_argument('-bid','--bidir',help="choices: [YES,NO]")
    commandLineArgument.add_argument('-t','--test',type=int,help="choices: [0,1]")
    commandLineArgument.add_argument('-att','--attention',type=int,help="choices: [0,1]")
    commandLineArgument.add_argument('-ht','--heat',type=int,help="choices: [0,1]")
    commandLineArgument.add_argument('-f','--font',help="Font of the language chosen to generate the heatmap")

    return commandLineArgument.parse_args()