import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('ChatGpt_Reviews.csv')
data['Review'] = data['Review'].fillna("")
data['Ratings'] = data['Ratings'].fillna(3)
data = data[data['Ratings'] != 3]
data['Sentiment'] = data['Ratings'].apply(lambda x: 1 if x >= 4 else 0)

X = data['Review']
y = data['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26)

vectorizer = CountVectorizer(max_features=3000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

def predict_sentiment(review_text):
    review_vec = vectorizer.transform([review_text])
    prediction = model.predict(review_vec)
    return "Positive" if prediction[0] == 1 else "Negative"

example_review = input("Enter you review: ")
predicted_sentiment = predict_sentiment(example_review)
print(f"Predicted Sentiment: {predicted_sentiment}")