
import joblib
import re
import string
from nltk import tokenize
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline


class eng_news_ml_model_pred():
    def __init__(self) -> None:
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


    def import_module(self):
        global LR,DT,GBC,RFC,XGB,LGB
        # ../models
        LR = joblib.load('English_news/models/LR.pkl')
        DT = joblib.load('English_news/models/DT.pkl')
        GBC = joblib.load('English_news/models/GBC.pkl')
        RFC = joblib.load('English_news/models/RFC.pkl')
        XGB = joblib.load('English_news/models/xgb.pkl')
        LGB = joblib.load('English_news/models/LGB.pkl')

    def predictions(self, text):
        text = self.wordopt(text)
        self.import_module()
        self.limetesting(tokens= text)
        

    def limetesting(self,tokens):
        vectorization = joblib.load('English_news/models/vectorization.pkl')
        class_names=["fake", "real"]
        explainer = LimeTextExplainer(class_names=class_names)
        tfidf  = vectorization
        
        print('#LR--------------------------------')
        
        c_lr = make_pipeline(tfidf, LR)
        exp_lr = explainer.explain_instance(tokens, c_lr.predict_proba, num_features=10)
        exp_lr.show_in_notebook(text=True)
        
        print('#DT--------------------------------')
        
        
        c_dt = make_pipeline(tfidf, DT)
        exp_dt = explainer.explain_instance(tokens, c_dt.predict_proba, num_features=10)
        exp_dt.show_in_notebook(text=True)
        
        print('#GBC--------------------------------')
        
        
        c_gbc = make_pipeline(tfidf, GBC)
        exp_gbc = explainer.explain_instance(tokens, c_gbc.predict_proba, num_features=10)
        exp_gbc.show_in_notebook(text=True)
        
        print('#RFC--------------------------------')
        
        
        c_rfc = make_pipeline(tfidf, RFC)
        exp_rfc = explainer.explain_instance(tokens, c_rfc.predict_proba, num_features=10)
        exp_rfc.show_in_notebook(text=True)
        
        print('#LGB--------------------------------')
        
        
        c_lgb = make_pipeline(tfidf, LGB)
        exp_lgb = explainer.explain_instance(tokens, c_lgb.predict_proba, num_features=10)
        exp_lgb.show_in_notebook(text=True)

# a = eng_news_ml_model_pred()
# a.predictions("I am a fake news")