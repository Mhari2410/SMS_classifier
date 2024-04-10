import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the Porter stemmer
ps = PorterStemmer()

# Define a function to preprocess the text
def transform_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)
    # Initialize an empty list to store preprocessed words
    y = []
    # Iterate through each word in the text
    for i in text:
        # Remove non-alphanumeric characters
        if i.isalnum():
            # Remove stopwords and punctuation, and apply stemming
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(ps.stem(i))  # Append the stemmed word to the list
    # Return the preprocessed text as a string
    return " ".join(y)

# Load the TF-IDF vectorizer and the model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Set the title of the Streamlit app
st.title("Email/SMS Spam Classifier")

# Create a text area for user input
input_sms = st.text_area("Enter the message")

# Check if the 'Predict' button is clicked
if st.button('Predict'):
    try:
        # Preprocess the input message
        transformed_sms = transform_text(input_sms)
        # Vectorize the preprocessed message
        vector_input = tfidf.transform([transformed_sms])
        # Make a prediction
        result = model.predict(vector_input)[0]
        # Display the prediction result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    except Exception as e:
        # Handle any exceptions and display an error message
        st.error(f"An error occurred: {e}")
