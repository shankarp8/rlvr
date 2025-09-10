from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen3-0.6B"

save_dir = "/home/sp2583/rlvr/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_dir)

model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained(save_dir)

print(f"Model and tokenizer saved to {save_dir}")