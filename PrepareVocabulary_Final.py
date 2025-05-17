from PrepareVocabulary_Initialize import initializeVocabularyDictionaries
from PrepareVocabulary_Create import createVocabulary

'''class to prepare the vocabulary of the dataset'''
class PrepareVocabulary:

    '''constructor to intialize the class parameters'''
    def __init__(self):

        '''define the start token, end token and padding token'''
        self.startOfSequenceToken="~"
        self.endOfSequenceToken="%"
        self.paddingToken="`"
        self.startOfSequenceIndex=0
        self.endOfSequenceIndex=1
        self.paddingIndex=2

        '''current vocabulary size is 3 (start token, end token, padding token)'''
        self.vocabularySizeForEnglish=3
        self.vocabularySizeForBengali=3

        self.charToIndexDictForEnglish=dict()
        self.indexToCharDictForEnglish=dict()
        self.charCounterForEnglish=dict()

        self.charToIndexDictForBengali=dict()
        self.indexToCharDictForBengali=dict()
        self.charCounterForBengali=dict()

        '''initialize the base vocabulary tokens '''
        self.initializeVocabularyDictionaries()

    ''' attach the split methods'''
    initializeVocabularyDictionaries = initializeVocabularyDictionaries
    createVocabulary = createVocabulary