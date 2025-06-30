import streamlit as st
import pickle
import string
import nltk

# Explicitly tell NLTK where to find the data folder
nltk.data.path.append('./nltk_data')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# Initialize the Porter Stemmer
ps = PorterStemmer()

# --- Preprocessing Function ---
def transform_text(text):
    
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english') and word not in string.punctuation]
    
    stemmed_tokens = [ps.stem(word) for word in tokens]
    
    return " ".join(stemmed_tokens)

# --- Model and Vectorizer Loading (with caching) ---
@st.cache_resource
def load_vectorizer():
    with open('vectorizer.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

# Load the resources
tfidf = load_vectorizer()
model = load_model()

# --- Streamlit App Interface ---
st.title("📧 Email / SMS Spam Classifier")
st.write("Enter a message below to check if it's spam or not.")

input_sms = st.text_area("Enter the message", height=150)

if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a message to classify.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        
        # 3. Predict
        result = model.predict(vector_input)[0]
        
        # 4. Display
        if result == 1:
            st.error("This message is likely Spam.")
        else:
            st.success("This message is Not Spam.")