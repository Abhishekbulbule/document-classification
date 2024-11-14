from django.db import models

# Create your models here.
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load or train your model here
model = joblib.load('D:/Abhishek/document-classification/backend/model.joblib')

def classify_document(document): 
    vectorizer = TfidfVectorizer(stop_words='english') 
    X = vectorizer.transform([document]) 
    category = model.predict(X)[0] 
    return category
