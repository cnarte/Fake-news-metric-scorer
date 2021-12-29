# from nltk import text
from transformers import pipeline
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class  hindi_news_zero_shot_model_pred():
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
        
        word_tokens = word_tokenize(text)
        filtered_sentence = ''

        for w in word_tokens:
            if  len(filtered_sentence) == 0:
                filtered_sentence += w
            elif  len(filtered_sentence) != 0:
                filtered_sentence += ' '+w
        
        
        return filtered_sentence

    def predict(self, text):
        candidate_labels = ["real", "fake"]
        question = "यह समाचार लेख {} है "
        text = self.wordopt(text)
        return self.classifier(text,candidate_labels,hypothesis_template =question)

if __name__ == "__main__":

    test_obj = hindi_news_zero_shot_model_pred()

    print(test_obj.predict(text= "हाल ही में एक हिन्दू महिला को एक कश्मीरी मुस्लिम लड़की को सहरी खिलाने पर आरएसएस के गुंडों ने पीटा| भारत आरएसएस के कब्जे में है| उनका मीडिया, लॉ एनफोर्समेंट, जुडिशरी और राष्ट्रिय सुरक्षा पर पूरा नियंत्रण है| क्या यह देश स्थिर रहेगा यदि ऐसा ही चलता रहा तो?"))