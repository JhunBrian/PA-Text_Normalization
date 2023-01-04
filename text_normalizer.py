import pandas as pd
import re
import json
import nltk
from textblob import TextBlob
from tqdm.notebook import tqdm

from warnings import filterwarnings as fws
fws('ignore')
    
class Normalizer:
    def __init__(self, data):
        self.data = data.dropna()
        
    def wrap_tqdm(self, function, desc):
        corrected = []
        for serie in tqdm(self.data, desc=desc):
            correct = function(serie)
            corrected.append(correct)
        return pd.Series(corrected)
        
    def remove_punctuation(self):
        punc_x_lower = lambda x : re.sub(r'[^\w\s]', '', x.lower())
            
        self.data = self.wrap_tqdm(punc_x_lower, 'Removing Punctuations')
        return self.data
    
    def expand_contractions(self):
        def expand_sentence(sentence):
            with open('contractions.json') as f:
                contraction = json.load(f)
            contraction_map = {k.lower():v for k,v in contraction.items()}
            
            words = sentence.split()
            expanded_words = []
            for word in words:
                if word in contraction_map:
                    expanded_words.append(contraction_map[word])
                else:
                    expanded_words.append(word)
            return " ".join(expanded_words)
        
        self.data = self.wrap_tqdm(expand_sentence, 'Expanding Contraction')
        return self.data
    
    def remove_stop_words(self):
        stop_words = set(nltk.corpus.stopwords.words('english'))
        stop_word_func = lambda x: ' '.join([word for word in x.split() if word not in stop_words])
            
        self.data = self.wrap_tqdm(stop_word_func, 'Removing Stopwords')
        return self.data
    
    def remove_repeating_characters(self):
        def remove_repeats(text):
            return re.sub(r'(\b\w+)\1+\b', r'\1', text)
            
        self.data = self.wrap_tqdm(remove_repeats, 'Removing Repeating Characters')
        return self.data
    
    def correct_spelling(self):
        def correct_text(sentence):
            corrected_words = []
            for word in sentence.split():
                corrected_word = str(TextBlob(word).correct())
                corrected_words.append(corrected_word)
            return ' '.join(corrected_words)
            
        self.data = self.wrap_tqdm(correct_text, 'Correcting Spelling')
        return self.data
    
    def normalize(self):
        self.remove_punctuation()
        self.expand_contractions()
        self.remove_stop_words()
        self.remove_repeating_characters()
        self.correct_spelling()
        
        return self.data