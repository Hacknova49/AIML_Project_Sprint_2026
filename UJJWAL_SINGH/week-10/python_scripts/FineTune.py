from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import os
from safetensors.torch import save_file

MODEL_NAME = "gpt2-medium"  # 355M parameter model
DATASET_PATH = "./dataset/short_brand_description_prepared.jsonl"
OUTPUT_DIR = "./slogan_generator_quantized"

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("json", data_files={"train": DATASET_PATH})["train"]

def preprocess(example):
    full_text = f"{example['prompt']}{example['completion']}"
    return tokenizer(
        full_text,
        truncation = True,
        max_length=128,
        padding="max_length",
        return_tensors = "pt"
    )

tokenized_dataset = dataset.map(preprocess,
    batched=False,
    remove_columns=dataset.column_names
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size =2,
    gradient_accumulation_steps =4,  # compensating for small batch size
    num_train_epochs =7,
    learning_rate =1.5e-5,
    warmup_steps=150,
    fp16 =True if torch.cuda.is_available() else False,
    logging_steps=50,
    save_strategy="epoch",
    optim="adamw_torch",
    report_to="none"        #disabled external logging
)

data_collator = DataCollatorForLanguageModeling(     #pad the tokenized inputs to the same length within each batch
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

print("Training started...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Model saved!")


print("Applying dynamic quantization to linear layers...")
# dynamically quantize(convert) the model linear layers to use int8
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
# override the quantized lm_head with the original full precision lm_head
# so that the final state_dict contains "lm_head.weight" (required for loading)
quantized_model.lm_head = model.lm_head

quantized_model.cpu()


print("Converting remaining FP32 parameters to FP16...")
state_dict = quantized_model.state_dict()
new_state_dict = {}
for key, tensor in state_dict.items():
    if isinstance(tensor, torch.Tensor):
        if tensor.dtype == torch.float32:
            new_state_dict[key] = tensor.half()    #to reduce size further
        else:
            new_state_dict[key] = tensor
    else:
        print(f"Skipping non-tensor key: {key}")


safetensors_path = os.path.join(OUTPUT_DIR, "model.safetensors")
save_file(new_state_dict, safetensors_path)
print(f"quantized model saved to {safetensors_path}")

tokenizer.save_pretrained(OUTPUT_DIR)
model.config.to_json_file(os.path.join(OUTPUT_DIR, "config.json"))
print("tokenizer and model configuration saved")