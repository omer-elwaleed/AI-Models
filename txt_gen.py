# Simple completion
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)


# Summarization example
text = """NASA's Perseverance rover successfully landed on Mars as part of the Mars Exploration Program.
It is designed to search for signs of ancient life, collect rock samples, and prepare for future missions."""
summary = generator(f"Summarize: {text}", max_length=50, min_length=20, do_sample=False)
print(summary[0]["generated_text"])