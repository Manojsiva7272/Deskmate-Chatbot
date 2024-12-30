from flask import Flask, request, jsonify, render_template  # Ensure render_template is imported
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk

app = Flask(__name__)
CORS(app)

# Download necessary resources
nltk.download('stopwords')
nltk.download('wordnet')

# Define stopwords, stemmer, and lemmatizer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    words = text.lower().split()
    processed_words = [lemmatizer.lemmatize(stemmer.stem(word)) for word in words if word not in stop_words]
    return ' '.join(processed_words)

# Load CSV data
file_path = r"C:\Users\Manoj\OneDrive\Pictures\Deskmate\data.csv"
data = pd.read_csv(file_path)
data = data.dropna(subset=['Question', 'Answer'])
data['Processed_Question'] = data['Question'].apply(preprocess_text)

processed_questions = data['Processed_Question'].tolist()
answers = data['Answer'].tolist()

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(lowercase=True)
question_vectors = tfidf_vectorizer.fit_transform(processed_questions)

# Function to get the best match for a user query
def get_answer(user_query, threshold=0.3):
    user_query_processed = preprocess_text(user_query)
    query_vector = tfidf_vectorizer.transform([user_query_processed])
    similarities = cosine_similarity(query_vector, question_vectors)
    max_similarity_index = similarities.argmax()
    max_similarity_score = similarities[0, max_similarity_index]

    if max_similarity_score >= threshold:
        return answers[max_similarity_index]
    return "I'm sorry, I couldn't find an answer to your question. Please try rephrasing."

# Function to greet the user
def greet_user():
    greetings = [
        "Hello! How can I assist you today?",
        "Hi there! How can I help you today?",
        "Hey! What can I do for you today?",
        "Greetings! How may I assist you?"
    ]
    return greetings[0]  # Choose the first greeting for simplicity

# API Endpoint for chatbot response
@app.route('/get-response', methods=['POST'])
def get_response():
    user_query = request.json.get('query', '')
    
    if user_query.strip() == "":
        response = greet_user()  # Greet the user if the query is empty or if it’s the first interaction
    else:
        response = get_answer(user_query)
    
    return jsonify({'response': response})

@app.route('/')
def index():
    return render_template('index.html')  # Save your HTML file as 'templates/index.html'

if __name__ == '__main__':
    app.run(debug=True)
