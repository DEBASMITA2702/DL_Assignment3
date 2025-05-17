'''creates vocabulary of each word in the dataset'''
def createVocabulary(self, dataset):
    '''
        Parameters:
            dataset : dataset on which to create the vocabulary
        Returns :
            None
        Function:
            creates vocabulary of each word in the dataset
    '''

    '''iterate over the entire dataset'''
    for each_pair in dataset:
        english_word=each_pair[0]
        bengali_word=each_pair[1]

        '''create vocabulary for the source language'''
        for one_char in english_word:
            '''if the character is not already recorded then add it to the dictionary'''
            if one_char not in self.charToIndexDictForEnglish:
                self.charToIndexDictForEnglish[one_char]=self.vocabularySizeForEnglish
                self.charCounterForEnglish[one_char]=1
                self.indexToCharDictForEnglish[self.vocabularySizeForEnglish]=one_char
                self.vocabularySizeForEnglish+=1
            else:
                self.charCounterForEnglish[one_char]+=1
        
        '''create vocabulary for the target language'''
        for one_char in bengali_word:
            '''if the character is not already recorded then add it to the dictionary'''
            if one_char not in self.charToIndexDictForBengali:
                self.charToIndexDictForBengali[one_char]=self.vocabularySizeForBengali
                self.charCounterForBengali[one_char]=1
                self.indexToCharDictForBengali[self.vocabularySizeForBengali]=one_char
                self.vocabularySizeForBengali+=1
            else:
                self.charCounterForBengali[one_char]+=1