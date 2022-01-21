import streamlit as st 
# from English_news.models import ml_models
from Hindi_news.models import transformer_zero_shot_model_hindi
from English_news.models.transformer_zero_shot_model_eng import eng_news_zero_shot_model_pred
import plotly.express as px

@st.cache(allow_output_mutation=True)
def load_model():
    eng_model = eng_news_zero_shot_model_pred()
    hindi_model = transformer_zero_shot_model_hindi.hindi_news_zero_shot_model_pred()
    return eng_model ,hindi_model
    
if __name__ == "__main__":
    st.title("Fake News Classification")
    #    st.write("")
    st.info("Zeroshot model loaded")
    st.subheader("Input the News content below")
    sentence = st.text_area( "Enter news artcle here",height=200)
    predict_btt_eng = st.button("Predict English")
    predict_btt_hindi = st.button("Predict Hindi")
    eng_model, hindi_model = load_model()

    if predict_btt_eng:
        prediction = eng_model.predict(sentence)
        fig = px.bar(x=prediction['labels'],y= prediction["scores"])
        st.plotly_chart(fig, use_container_width=True)
    if predict_btt_hindi:
        prediction = hindi_model.predict(sentence)
        fig = px.bar(x=prediction['labels'],y= prediction["scores"])
        st.plotly_chart(fig, use_container_width=True)
    

    