from textblob import TextBlob

text = input("Enter a hotel review: ")

blob = TextBlob(text)
sentiment = blob.sentiment.polarity

if sentiment > 0:
    print("positive review")
elif sentiment < 0:
    print("negative review")
else:
    print("neutral review")