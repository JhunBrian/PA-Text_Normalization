import re
import string
import nltk
import pandas as pd
import json

import contractions
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.corpus import words

class Normalizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = stopwords.words('english')
        self.english_words = set(words.words())

    def lowercase(self, data):
        return data.lower()
    
    def remove_punctuations_and_emojis(self, data):

        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)

        data = data.translate(str.maketrans('', '', string.punctuation))
        data = re.sub(emoji_pattern, '', data)
        return data
    
    def expand_sentence(self, data):
        contraction_map = {k.lower():v.lower() for k,v in contractions.contractions_dict.items()}

        words = data.split()
        expanded_words = []
        for word in words:
            if word in contraction_map:
                expanded_words.append(contraction_map[word])
            else:
                expanded_words.append(word)
        return " ".join(expanded_words)
    
    def remove_stop_words(self, data):
        return " ".join([word for word in data.split() if word not in self.stop_words])
    
    def remove_non_english_words(self, data):
        return " ".join([word for word in data.split() if word in self.english_words])
    
    def lemmatize(self, data):
        return " ".join([self.lemmatizer.lemmatize(word) for word in data.split()])
    
    def stem(self, data):
        return " ".join([self.stemmer.stem(word) for word in data.split()])
    
#     def word_tokenize(self, data):
#         return data.split()

    def normalize(self, data):
        data = data.apply(self.lowercase)
        data = data.apply(self.remove_punctuations_and_emojis)
        data = data.apply(self.expand_sentence)
        data = data.apply(self.remove_stop_words)
        data = data.apply(self.remove_non_english_words)
        data = data.apply(self.lemmatize)
        data = data.apply(self.stem)
#         data = data.apply(self.word_tokenize)

        return data

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# import pandas as pd

class Encoder:
    def __init__(self, data):
        self.data = data
    
    def bow(self, n_gram=(1,1)):
        vectorizer = CountVectorizer(min_df=1, ngram_range=n_gram)
        bow = vectorizer.fit_transform(self.data)
        return pd.DataFrame(bow.toarray(), columns=vectorizer.get_feature_names_out())
    
    def tf_idf(self):
        vectorizer = TfidfVectorizer(norm='l2', smooth_idf=True, use_idf=True)
        tfidf = vectorizer.fit_transform(self.data)
        return pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names_out())
