from nltk import text
from transformers import pipeline
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class  eng_news_zero_shot_model_pred():
    def __init__(self) -> None:
        self.classifier = pipeline("zero-shot-classification",model = "joeddav/xlm-roberta-large-xnli")
        super().__init__()
    def wordopt(self, text):
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub("\\W"," ",text) 
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_sentence = ''

        for w in word_tokens:
            if w not in stop_words and len(filtered_sentence) == 0:
                filtered_sentence += w
            elif w not in stop_words and len(filtered_sentence) != 0:
                filtered_sentence += ' '+w
        
        
        return filtered_sentence

    def predict(self, text):
        candidate_labels = ["real", "fake"]
        question = "This news article is {}"
        text = self.wordopt(text)
        return self.classifier(text,candidate_labels,hypothesis_template =question)

test_obj = eng_news_zero_shot_model_pred()

print(test_obj.predict(text= "This is fake news"))