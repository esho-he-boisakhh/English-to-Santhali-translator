
import pickle


import nltk

# Download the required packages
#nltk.download('punkt')  # This is for tokenizing words
#nltk.download('stopwords')  # This is for stopwords
#nltk.download('punkt_tab')  # This is to fix the missing punkt_tab


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#ei package gulo shobkota lagbe kina bujhte parchhi na ami, but onno ra use korchhe dekhe amio korlam

#program start

santhali_stopwords = ["to", "and", "is", "the", "in", "for", "of", "a", "on", "with"] 

# Loading dataset
data = pd.read_csv(r'D:\Prabrisha\Program College\vaani\Dataset1.csv')

# Displaying the first few rows of the dataset
print(data.head())


def remove_stopwords(sentence):
    """
    santhali_stopwords = ["to", "and", "is", "the", "in", "for", "of", "a", "on", "with"]  # TBA
    
    #Removes stopwords from the given sentence.

    Returns:
    list: A list of words from the sentence with stopwords removed.
    """
    # Tokenize the sentence into words
    tokens = word_tokenize(sentence)
    
    # Filter out the stopwords
    filtered_tokens = [word for word in tokens if word.lower() not in santhali_stopwords]
    
    return filtered_tokens


def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Convert to lowercase
    tokens = text.lower()
    
    # Remove punctuation
    #tokens = [word for word in tokens if word.isalpha()]
    
    # Remove stopwords
    # Load English stopwords
    english_stopwords = set(stopwords.words('english'))  
    # Combine stopwords
    combined_stopwords = english_stopwords.union(santhali_stopwords)

    # Remove stopwords
    filtered_tokens = remove_stopwords(text)
    
    return ' '.join(filtered_tokens)

# Applying preprocessing to dataset
data['processed_text'] = data['word'].apply(preprocess_text)

print(data.head())


# Initializing TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the data
X = vectorizer.fit_transform(data['processed_text'])

print(X.shape)  # Output the shape of the TF-IDF matrix


# Separation of dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, data['item'], test_size=0.2, random_state=42)

# Model training starting here

# 1. Model Initalization and training

model = LogisticRegression()
model.fit(X_train, y_train)

# 2. Model learns to make predictions
y_pred = model.predict(X_test)

# 3. Model check
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    features = vectorizer.transform([processed_text])
    prediction = model.predict(features)
    return prediction[0]

# Example usage
new_text = "Insert Santhali word/sentence"
print(f'Prediction: {predict_sentiment(new_text)}')

# Assuming 'model' is your trained model object
with open('apollo_3.pkl', 'wb') as model_file:
   pickle.dump(model, model_file)

# Save the vectorizer
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully!")
