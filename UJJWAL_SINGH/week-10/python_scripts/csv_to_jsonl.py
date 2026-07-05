import pandas as pd
import json

df = pd.read_csv("./dataset/short_brand_description.csv")

with open("short_brand_description.jsonl", "w") as f:
    for brand,slogan,desc in zip(df["Company"],df["Slogan"],df["Shortened_Description"]):
        entry = {
            "prompt":f"Brand: {brand} | Description: {desc}\nSlogan:",
            "completion":f" {slogan}\n"
        }
        f.write(json.dumps(entry) + "\n")

print("converted CSV to JSONL and saved!")