import pickle

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


X = ["hello world", "free money now", "call me", "win prize"]
y = [0, 1, 0, 1]

tfidf = TfidfVectorizer()
X_vector = tfidf.fit_transform(X)

model = MultinomialNB()
model.fit(X_vector, y)


with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Saved successfully")