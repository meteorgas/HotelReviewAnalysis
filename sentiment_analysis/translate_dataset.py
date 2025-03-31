import pandas as pd
from googletrans import Translator

# Load dataset
df = pd.read_csv("../review_dataset/chennai_reviews.csv")

# Create a copy of the first 100 rows
df_subset = df.head(100).copy()

# Translate only this subset
translator = Translator()
df_subset["Review_Text_TR"] = df_subset["Review_Text"].apply(
    lambda x: translator.translate(str(x), src="en", dest="tr").text
)

# Save only the translated 100 rows
df_subset.to_csv("translated_first_1001.csv", index=False)
