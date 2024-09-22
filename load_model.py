import pickle

# Load the model
with open('apollo_3.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

print("Model loaded successfully!")
