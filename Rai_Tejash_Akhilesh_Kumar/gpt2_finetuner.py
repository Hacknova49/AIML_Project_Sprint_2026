import torch
import evaluate
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling

def load_review_data():
    sample_review = "This product is amazing! I use it every day. Highly recommend to anyone looking for quality."
    dataset = Dataset.from_dict({"text": [sample_review] * 500})
    return dataset

def execute_pipeline():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    dataset = load_review_data()
    
    def tokenize_func(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=32)
        
    tokenized_dataset = dataset.map(tokenize_func, batched=True)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir="./product_review_model",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_steps=50,
        save_strategy="no",
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    trainer.train()
    
    prompt = "This product is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    attention_mask = tokenizer(prompt, return_tensors="pt").attention_mask.to(model.device)
    
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=30,
        do_sample=True,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
    
    bleu_metric = evaluate.load("sacrebleu")
    results = bleu_metric.compute(
        predictions=[generated_text], 
        references=[["This product is amazing! I use it every day. Highly recommend to anyone looking for quality."]]
    )
    print(results["score"])

if __name__ == "__main__":
    execute_pipeline()