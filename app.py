import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import word_tokenize # We will still use this, but after loading the data
from nltk.stem.porter import PorterStemmer

# --- NLTK Data Loading: The Manual, Bulletproof Way ---

# Define the path to the Punkt tokenizer pickle file.
# This file is inside the nltk_data folder you committed to Git.
PUNKT_PICKLE = "nltk_data/tokenizers/punkt/PY3/english.pickle"

ps = PorterStemmer()
# Load the tokenizer manually
try:
    with open(PUNKT_PICKLE, 'rb') as f:
        tokenizer = pickle.load(f)
except FileNotFoundError:
    # This is a fallback for local testing if the folder structure is different.
    # On Render, the above `try` block should succeed.
    st.error("Punkt tokenizer not found! Please ensure 'nltk_data' directory is available.")
    # Download if all else fails, though this part shouldn't run on a successful deploy
    import nltk
    nltk.download('punkt')
    with open(PUNKT_PICKLE, 'rb') as f:
        tokenizer = pickle.load(f)

# Also ensure stopwords are available
try:
    stopwords.words('english')
except LookupError:
    import nltk
    nltk.download('stopwords')


# Initialize the Porter Stemmer
ps = PorterStemmer()


# --- Preprocessing Function (Modified) ---
def transform_text(text):
    """Performs text preprocessing using a manually loaded tokenizer."""
    text = text.lower()
    
    # Use the standard word_tokenize function, which will now find the data
    # because the loader has already confirmed its presence.
    # If word_tokenize still fails, we use the manually loaded one as a backup.
    try:
        tokens = word_tokenize(text)
    except LookupError:
        # This is a robust fallback
        tokens = tokenizer.tokenize(text)

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