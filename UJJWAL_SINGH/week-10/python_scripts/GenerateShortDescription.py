import pandas as pd
import requests
from tqdm import tqdm
import spacy


nlp = spacy.load("en_core_web_sm")  # using this to generate short description by taking keywords from descriptions generated

df = pd.read_csv("./dataset/sloganlist.csv")

brand_names = df["Company"]#.unique()

def get_brand_description(brand):
    response = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{brand}")  # fetching descriptions from wikipedia

    if response.status_code==200:
        return response.json().get("extract", "No description available")  # get the description from "extract" key in the json returned

    return "No description available"

df["Description"] = [get_brand_description(brand) for brand in tqdm(brand_names, desc="fetching descriptions")]  # using tqdm for seeing progress in real time

def summarize_desc(text):
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "ADJ"]]  #select those words(keywords) which are a noun, proper noun or adjective
    return " ".join(keywords[:12]) #limiting to 12 keywords only

df["Shortened_Description"] = [summarize_desc(desc) for desc in tqdm(df["Description"], desc="procesing description")]

df.to_csv("short_brand_description.csv", index=False)

print("Saved the final description to csv file!")