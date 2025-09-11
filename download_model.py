from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-3B-Instruct"

save_dir = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_dir)

model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained(save_dir)

print(f"Model and tokenizer saved to {save_dir}")