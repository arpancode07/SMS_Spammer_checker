import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
#nltk.download('punkt_tab')

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Detection")

input_message = st.text_area("Enter the message for detection")

if st.button("Predict"):
    # 1. Preprocess Text
    def transform_text(text):
        text = text.lower()
        text = nltk.word_tokenize(text)

        ans = []

        for i in text:
            if i.isalnum():
                ans.append(i)

        text = ans[:]
        ans.clear()
        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                ans.append(i)

        text = ans[:]
        ans.clear()
        for i in text:
            ans.append(ps.stem(i))

        return " ".join(ans)


    transformed_message = transform_text(input_message)

    # 2. Vectorize the message
    vectorize_message = tfidf.transform([transformed_message])

    # 3. Prediction phase
    result = model.predict(vectorize_message)[0]

    # 4. Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
