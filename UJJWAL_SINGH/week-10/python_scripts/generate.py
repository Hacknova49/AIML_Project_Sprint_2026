from transformers import pipeline, GPT2Tokenizer
import torch
import spacy

nlp = spacy.load("en_core_web_sm")
def summarize_desc(text):
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "ADJ"]]  #select those words(keywords) which are a noun, proper noun or adjective
    return " ".join(keywords[:12]) #limiting to 12 keywords only

def generate_brand_slogans(brand, description, industry, tone="playful", num=5, liked_slogan=None):
    TONE_PRESETS = {
        "playful": {"temperature": 0.95, "top_p": 0.95, "repetition_penalty": 1.2},
        "bold": {"temperature": 0.8, "top_p": 0.9, "repetition_penalty": 1.45},
        "minimalist": {"temperature": 0.6, "top_p": 0.8, "repetition_penalty": 1.5},
        "luxury": {"temperature": 0.7, "top_p": 0.85, "repetition_penalty": 1.35},
        "classic": {"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.25}
    }

    tokenizer = GPT2Tokenizer.from_pretrained("./SloganGenerator")
    generator = pipeline(
        "text-generation",
        model="./slogan_generator_medium",
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

    description_summarized = summarize_desc(description)

    if liked_slogan:
        # if a liked slogan is provided
        prompt1 = (
            f"Create {industry} industry brand slogans similar to: '{liked_slogan}'\n"
            f"Brand: {brand}\n"
            f"Key Attributes: {description_summarized}\n"
            "Slogan:"
        )
        
        prompt2 = (
            f"Generate slogans in the same spirit as '{liked_slogan}, for {industry} industry'\n"
            f"Brand: {brand}\n"
            f"Details: {description_summarized}\n"
            "Slogan:"
        )
    else:
        # attribute focused
        prompt1 = (
            f"Create a {industry} brand slogan that's catchy and unique.\n"
            f"Brand: {brand}\n"
            f"Key Attributes: {description_summarized}\n"
            "Slogan:"
        )
        
        # impact focused
        prompt2 = (
            f"Write high-impact marketing slogans for a {industry} brand.\n"
            f"Brand: {brand}\n"
            f"Key Attributes: {description_summarized}\n"
            "Slogan:"
        )

    # generating parameters
    gen_params = {
        "max_new_tokens": 20,
        "temperature": TONE_PRESETS[tone]["temperature"],
        "top_p": TONE_PRESETS[tone]["top_p"],
        "top_k": 30,
        "repetition_penalty": TONE_PRESETS[tone]["repetition_penalty"],
        "num_return_sequences": num,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id
    }

    # generate from both prompts
    outputs1 = generator(prompt1, **gen_params)
    outputs2 = generator(prompt2, **gen_params)
    
    # processing and combining results
    slogans = []
    for output_group in [outputs1, outputs2]:
        for o in output_group:
            raw = o['generated_text'].split("Slogan:")[-1].strip()
            clean = raw.split("\n")[0].replace('"', '').replace('(', '').split(".")[0]
            if len(clean) > 4:
                slogans.append(clean)

    # return slogans[:num*2]  # Return requested number of unique slogans
    return list(set(slogans))[:num*2]  # remove duplicate and limit output



# original generation
original_slogans = generate_brand_slogans(
    "GreenWeave", 
    "A sustainable fashion brand.",
    industry="fashion",
    tone="playful"
)

print(original_slogans)

# user selects a slogan they like
liked_slogan = 'Your Style Starts Here!'

# generating similar slogans
similar_slogans = generate_brand_slogans(
    "GreenWeave", 
    "A sustainable fashion brand.",
    industry="fashion",
    tone="minimalist",
    liked_slogan=liked_slogan
)

print(similar_slogans)