from flask import Flask, request, jsonify
import pickle  # Assuming you saved your model using pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your trained model
with open('apollo_3.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def preprocess_text(text):
    # Add your preprocessing logic here
    return text.lower()  # Example: simple lowercase

def predict_translation(text):
    processed_text = preprocess_text(text)
    features = vectorizer.transform([processed_text])
    prediction = model.predict(features)
    return prediction[0]

@app.route('/', methods=["POST"])
def translate():
    data = request.json
    english_word = data.get('word')
    
    if not english_word:
        return jsonify({'error': 'No word provided'}), 400
    
    predicted_santhali = predict_translation(english_word)
    
    return jsonify({'santhali_word': predicted_santhali})

if __name__ == '__main__':
    app.run(debug=True)
