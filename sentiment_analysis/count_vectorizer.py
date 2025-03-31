from sklearn.feature_extraction.text import CountVectorizer

texts = [
    "I LOVE t this hotel",
    "This hotel is bad",
    "I hate this place"
]

vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
X = vectorizer.fit_transform(texts)

print("Vocabulary:", vectorizer.vocabulary_)

print("Vectors:\n", X.toarray())