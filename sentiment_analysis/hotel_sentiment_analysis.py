# We’re teaching a computer how to look at a hotel review and guess if it’s positive (1) or negative (0).
# This is called sentiment classification, and we use a method called machine learning to do it.

#📌 We're bringing in a tool called **Pandas**. It's used to store and work with **tables of data**, like Excel sheets in Python.
import pandas as pd

#📌 This function splits your data into two parts:
# 	•	One for training the model (letting it learn).
# 	•	One for testing how well it learned.
from sklearn.model_selection import train_test_split

# 📌 Text is **just words**, but computers only understand **numbers**.
# `CountVectorizer` is a tool that **turns each sentence into a list of word counts** (i.e., numbers the computer can understand).
#
# For example:
# ```text
# "The hotel was clean"
# → ["hotel", "clean"] → [1, 0, 1, ...]
from sklearn.feature_extraction.text import CountVectorizer

# 📌 This is the **machine learning model** we’re using.
# It’s called **Naive Bayes**, and it's a simple algorithm that's good at understanding text like spam filters and reviews.
from sklearn.naive_bayes import MultinomialNB

# 📌 This helps us measure how well the model is doing — like a grade or a score.
from sklearn.metrics import accuracy_score

# 📌 This is the data we’re using to train and test the model.
# We have **6 reviews**, and each review is labeled:
# - `1` = **Positive**
# - `0` = **Negative**
#
# So we are telling the computer:
# > “Hey, here's a review and what kind of emotion it has. Learn from it.”
data = {
    'review': [
        "The hotel was amazing and clean",
        "I hated the room, it was dirty",
        "Very friendly staff and great service",
        "Terrible experience, never coming back",
        "Lovely place, felt like home",
        "Rude staff and awful food"
    ],
    'sentiment': [1, 0, 1, 0, 1, 0]
}

# 📌 We put our reviews into a data table (like a spreadsheet) using Pandas, so we can work with them more easily.
df = pd.DataFrame(data)

# 📌 We **split the data** into two parts:
# - **X_train**: Reviews for training
# - **X_test**: Reviews for testing
# - **y_train**: Correct answers for training
# - **y_test**: Correct answers for testing
# `test_size=0.3` means we’re using **30% of data for testing** (2 reviews) and 70% (4 reviews) for training.
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.3)

# Step 2: Convert text to numbers
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 3: Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 4: Predict and evaluate
predictions = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, predictions))

# Step 5: Try your own review
your_review = input("Write your own hotel review: ")
your_review_vec = vectorizer.transform([your_review])
prediction = model.predict(your_review_vec)

if prediction[0] == 1:
    print("😊 Positive review")
else:
    print("😠 Negative review")