import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 📌 STEP 1: Load data from your CSV file
# Make sure your file has columns: Review_Text, Sentiment
df = pd.read_csv("../review_dataset/chennai_reviews.csv")
df = df.rename(columns={"Review_Text": "review", "Sentiment": "sentiment"})

# Drop rows where review OR sentiment is missing
df = df.dropna(subset=["review", "sentiment"])

# Convert sentiment to int (forcefully)
df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")  # Converts non-numeric to NaN
df = df.dropna(subset=["sentiment"])  # Drop rows where sentiment became NaN
df["sentiment"] = df["sentiment"].astype(int)

# Keep only sentiments in desired range
df = df[df["sentiment"].isin([0, 1, 2, 3])]

# 📌 STEP 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.3)

# 📌 STEP 4: Convert text to numbers
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 📌 STEP 5: Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 📌 STEP 6: Predict and evaluate
predictions = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, predictions))

# 📌 STEP 7: Try your own review
your_review = input("Write your own hotel review: ")
your_review_vec = vectorizer.transform([your_review])
prediction = model.predict(your_review_vec)

if prediction[0] == 1:
    print("😊 Positive review")
else:
    print("😠 Negative review")