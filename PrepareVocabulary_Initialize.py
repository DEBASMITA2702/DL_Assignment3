'''initializes the vocabulary dictionaries'''
def initializeVocabularyDictionaries(self):
    '''
        Parameters:
            None
        Returns :
            None
        Function:
            Initializes the vocabulary dictionaries
    '''

    '''dictionary for source language'''
    self.charToIndexDictForEnglish[self.startOfSequenceToken]=self.startOfSequenceIndex
    self.charToIndexDictForEnglish[self.endOfSequenceToken]=self.endOfSequenceIndex
    self.charToIndexDictForEnglish[self.paddingToken]=self.paddingIndex

    self.indexToCharDictForEnglish[self.startOfSequenceIndex]=self.startOfSequenceToken
    self.indexToCharDictForEnglish[self.endOfSequenceIndex]=self.endOfSequenceToken
    self.indexToCharDictForEnglish[self.paddingIndex]=self.paddingToken

    '''dictionary for target language'''
    self.charToIndexDictForBengali[self.startOfSequenceToken]=self.startOfSequenceIndex
    self.charToIndexDictForBengali[self.endOfSequenceToken]=self.endOfSequenceIndex
    self.charToIndexDictForBengali[self.paddingToken]=self.paddingIndex

    self.indexToCharDictForBengali[self.startOfSequenceIndex]=self.startOfSequenceToken
    self.indexToCharDictForBengali[self.endOfSequenceIndex]=self.endOfSequenceToken
    self.indexToCharDictForBengali[self.paddingIndex]=self.paddingToken