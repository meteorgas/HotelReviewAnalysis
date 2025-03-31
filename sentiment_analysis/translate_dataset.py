import pandas as pd
import asyncio
from googletrans import Translator

df = pd.read_csv("../review_dataset/chennai_reviews.csv")
translator = Translator()

async def translate_text(text):
    try:
        return (await translator.translate(str(text), src='en', dest='tr')).text
    except:
        return ""

async def main():
    tasks = [translate_text(text) for text in df["Review_Text"].astype(str)]
    translated = await asyncio.gather(*tasks)
    df["Review_Text_TR"] = translated
    df.to_csv("../review_dataset/translated_chennai_reviews.csv", index=False)

asyncio.run(main())