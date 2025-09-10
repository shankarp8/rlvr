import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("/home/sp2583/rlvr/distill_qwen_1.5b", torch_dtype=torch.bfloat16).cuda()
tok = AutoTokenizer.from_pretrained("/home/sp2583/rlvr/distill_qwen_1.5b")
x = tok("hello", return_tensors="pt").to("cuda")
with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        print("sdpa/math start"); y = m(**x, labels=x["input_ids"]); print("sdpa/math ok")
        torch.cuda.synchronize()
        print("done")
