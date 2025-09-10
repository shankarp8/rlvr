from vllm import LLM, SamplingParams
llm = LLM(model="/home/sp2583/rlvr/distill_qwen_1.5b")  # same BASE_MODEL
print("ok: engine constructed")
out = llm.generate(["hello"], SamplingParams(max_tokens=8))
print("ok: generated", out[0].outputs[0].text)
