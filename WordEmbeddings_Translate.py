import torch
import torch.nn as nn

class WordEmbeddings:
    def translateWordToTensor(self, word, vocabulary, language):
        '''
            Parameters:
                word : word on which to create the embeddings
                vocabulary : vocabulary of the dataset
                language : language of the dataset
            Returns :
                trans : embedding of the word
            Function:
                Generates the embeddings
        '''
        tensorList = list()
        if language == "english":
            tensorList.append(
                vocabulary.charToIndexDictForEnglish[vocabulary.startOfSequenceToken]
            )
        else:
            tensorList.append(
                vocabulary.charToIndexDictForBengali[vocabulary.startOfSequenceToken]
            )

        for one_char in word:
            if language == "english":
                tensorList.append(vocabulary.charToIndexDictForEnglish[one_char])
            else:
                tensorList.append(vocabulary.charToIndexDictForBengali[one_char])

        if language == "english":
            tensorList.append(
                vocabulary.charToIndexDictForEnglish[vocabulary.endOfSequenceToken]
            )
        else:
            tensorList.append(
                vocabulary.charToIndexDictForBengali[vocabulary.endOfSequenceToken]
            )

        trans = torch.tensor(tensorList, dtype=torch.int64)
        return trans
