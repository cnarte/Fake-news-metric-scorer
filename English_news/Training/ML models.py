# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

# %%
df_fake = pd.read_csv("../input/fake-news-detection/Fake.csv")
df_true = pd.read_csv("../input/fake-news-detection/True.csv")
# %%
df_fake["class"] = 0
df_true["class"] = 1


# %%
df_fake_manual_testing = df_fake.tail(10)
for i in range(23480,23470,-1):
    df_fake.drop([i], axis = 0, inplace = True)
    
    
df_true_manual_testing = df_true.tail(10)
for i in range(21416,21406,-1):
    df_true.drop([i], axis = 0, inplace = True)

# %%
df_fake.shape, df_true.shape

# %%
df_fake_manual_testing["class"] = 0
df_true_manual_testing["class"] = 1

# %%
df_fake_manual_testing.head(10)

# %%
df_true_manual_testing.head(10)

# %%
df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis = 0)
df_manual_testing.to_csv("manual_testing.csv")

# %%
df_merge = pd.concat([df_fake, df_true], axis =0 )
#df_merge.head(10)

# %%
df_merge.columns

# %%
df = df_merge.drop(["title", "subject","date"], axis = 1)

# %%
df.head(10)

# %%
df.isnull().sum()

# %%
df = df.sample(frac = 1)

# %%
df.head()

# %%
df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)

# %%
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
def wordopt(text):
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

# %%
df["text"] = df["text"].apply(wordopt)

# %%
x = df["text"]
y = df["class"]

# %%
x.count()

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
#xv_train = vectorization.fit_transform(x_train)
#xv_test = vectorization.transform(x_test)
xv_train = vectorization.fit_transform(x)

# %%
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
#LR.fit(xv_train,y_train)
LR.fit(xv_train,y)
#pred_lr=LR.predict(xv_test)

# %%
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
# DT.fit(xv_train, y_train)
# pred_dt = DT.predict(xv_test)
DT.fit(xv_train,y)

# %%
from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier(random_state=0)
# GBC.fit(xv_train, y_train)
# pred_gbc = GBC.predict(xv_test)
GBC.fit(xv_train,y)

# %%
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(random_state=0)
# RFC.fit(xv_train, y_train)
# pred_rfc = RFC.predict(xv_test)
RFC.fit(xv_train,y)

# %%
pip install xgboost

# %%
from xgboost import XGBClassifier
xgb = XGBClassifier()
# xgb.fit(xv_train, y_train)
# pred_xgb = xgb.predict(xv_test)
xgb.fit(xv_train,y)

# %%
import lightgbm as lgb
LGB = lgb.LGBMClassifier()
# LGB.fit(xv_train, y_train)
# pred_lgb=LGB.predict(xv_test)
LGB.fit(xv_train,y)

# %%
pip install sklearn

# %%
import joblib

# %%
joblib.dump(vectorization, 'vectorization.pkl')

# %%
joblib.dump(LR, 'LR.pkl')
joblib.dump(DT, 'DT.pkl')
joblib.dump(GBC, 'GBC.pkl')
joblib.dump(RFC, 'RFC.pkl')
joblib.dump(xgb, 'xgb.pkl')
joblib.dump(LGB, 'LGB.pkl')

# %%
print(classification_report(y_test, pred_rfc))

# %%
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
class_names=['non authentic','authentic']
explainer = LimeTextExplainer(class_names=class_names)
tfidf  = vectorization

# %%
c = make_pipeline(tfidf, RFC)
#c = make_pipeline(tfidf, LGB)
#c = make_pipeline(tfidf, xgb)

# %%
x_test

# %%
idx = 35701#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# %%
exp = explainer.explain_instance(a.E[25], c.predict_proba, num_features=10)
exp1 = explainer.explain_instance(a.E[30], c.predict_proba, num_features=10)

# %%
print('Document id: %d' % idx)
print('Probability=', c.predict_proba([df.text[idx]])[0,1])
exp.as_list()

# %%
exp.show_in_notebook(text=True)
exp1.show_in_notebook(text=True)

# %%
def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)
    pred_xgb = xgb.predict(new_xv_test)
    pred_lgb = LGB.predict(new_xv_test)
    print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {} \nxgb Prediction: {} \nLBG Prediction: {}".format(output_lable(pred_LR[0]),
                                                                                                              output_lable(pred_DT[0]), 
                                                                                                              output_lable(pred_GBC[0]), 
                                                                                                              output_lable(pred_RFC[0]),
                                                                                                              output_lable(pred_xgb[0]),
                                                                                                              output_lable(pred_lgb[0])))
    print("#-------------------------------------------------------------------------------")

# %%
df_testvikas = pd.read_csv("../input/testvikas12/NLP_DATA - vikas.csv")

# %%
a = pd.DataFrame(df_testvikas['E'])

# %%
a['E'] =a['E'].apply(wordopt)

# %%
a["E"][0] = 'Khan was initially sent to the custody of NCB for questioning, but a Mumbai court later denied NCB plea for further extension of custody and sent Khan to 14-day judicial remand on October 7. His multiple bail petitions have so far been rejected.Besides Khan, the bail application of other accused Arbaaz Merchant and Munmun Dhamecha will also be heard on Wednesday.The court had posted the matter for October 20 after hearing arguments on October 13 and 14. Additional Session Judge VV Patil had said that he would pronounce his order on October 20.Khan, the son of actor Shah Rukh Khan, was among eight people arrested after an NCB raid on Cordelia, a holiday cruise ship anchored in Mumbai that was set to leave for Goa later that evening.'

# %%
for i in range(1):
    b = a['E'][i]
    #print(b)
    manual_testing(b)

# %%
#a['E'].apply(manual_testing)

# %%
def limetesting(idx):
    '''print('#LR------------------------------------------------------------------------------')
    
    c_lr = make_pipeline(tfidf, LR)
    exp_lr = explainer.explain_instance(a.E[idx], c_lr.predict_proba, num_features=10)
    exp_lr.show_in_notebook(text=True)
    
    print('#DT------------------------------------------------------------------------------')
    
    
    c_dt = make_pipeline(tfidf, DT)
    exp_dt = explainer.explain_instance(a.E[idx], c_dt.predict_proba, num_features=10)
    exp_dt.show_in_notebook(text=True)
    
    print('#GBC------------------------------------------------------------------------------')
    
    
    c_gbc = make_pipeline(tfidf, GBC)
    exp_gbc = explainer.explain_instance(a.E[idx], c_gbc.predict_proba, num_features=10)
    exp_gbc.show_in_notebook(text=True)'''
    
    print('#RFC------------------------------------------------------------------------------')
    
    
    c_rfc = make_pipeline(tfidf, RFC)
    exp_rfc = explainer.explain_instance(a.E[idx], c_rfc.predict_proba, num_features=10)
    exp_rfc.show_in_notebook(text=True)
    print(exp_rfc)
    '''print('#XGB------------------------------------------------------------------------------')
    
    
    c_xgb = make_pipeline(tfidf, xgb)
    exp_xgb = explainer.explain_instance(a.E[idx], c_xgb.predict_proba, num_features=10)
    exp_xgb.show_in_notebook(text=True)
    
    print('#LGB------------------------------------------------------------------------------')
    
    
    c_lgb = make_pipeline(tfidf, LGB)
    exp_lgb = explainer.explain_instance(a.E[idx], c_lgb.predict_proba, num_features=10)
    exp_lgb.show_in_notebook(text=True)'''
    
    
    

# %%
for i in range(100):
    limetesting(i)

# %%
#news = str(input())
#manual_testing(news)


