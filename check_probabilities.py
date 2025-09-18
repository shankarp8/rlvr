

import json, math, re
from typing import Optional, Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

VAL_PATH = "rlcr_pqa_0or1_validation.json"   
OUT_JSONL = "eval_qwen0or1_results.jsonl"

BASE_NAME = "Qwen/Qwen2.5-3B-Instruct"
FT_REPO  = "tanyagoyal-p/qwen3_0or1_2e-6"
FT_SUBFOLDER = "global_step_242/actor"      

MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.5
DO_SAMPLE = True
def _normalize_text(s) -> str:
    s = "" if s is None else str(s)
    return s.strip().lower()

def _try_float(x):
    try:
        if isinstance(x, str):
            x = x.replace(',', '').strip()
        return float(x)
    except Exception:
        return None

_ANSWER_FALLBACK_REGEXES = [
    r'(?is)\banswer\s*[:\-]\s*([A-D])\b',
    r'(?is)\banswer\s*[:\-]\s*(-?\d+(?:\.\d+)?)\b',
    r'(?is)\b([A-D])\b\s*(?:</?confidence>|$)',
    r'(?is)\b(-?\d+(?:\.\d+)?)\b\s*(?:</?confidence>|$)',
]

def _extract_answer_fallback(text: str) -> Optional[str]:
    for rx in _ANSWER_FALLBACK_REGEXES:
        m = re.search(rx, text)
        if m:
            return m.group(1).strip()
    return None

def _last_block(s: str, tag: str) -> Optional[str]:
    blocks = re.findall(fr'<{tag}>(.*?)</{tag}>', s, flags=re.DOTALL | re.IGNORECASE)
    return blocks[-1].strip() if blocks else None

def _first_block(s: str, tag: str) -> Optional[str]:
    blocks = re.findall(fr'<{tag}>(.*?)</{tag}>', s, flags=re.DOTALL | re.IGNORECASE)
    return blocks[0].strip() if blocks else None

def _count_blocks(s: str, tag: str) -> int:
    return len(re.findall(fr'<{tag}>(.*?)</{tag}>', s, flags=re.DOTALL | re.IGNORECASE))

def _assistant_segment(s: str) -> str:
    if "<|im_start|>assistant" in s:
        seg = s.rsplit("<|im_start|>assistant", 1)[-1]
        seg = seg.split("<|im_end|>", 1)[0]
        return seg
    return s

def _distill_segment(s: str) -> str:
    if '<｜Assistant｜>' in s:
        seg = s.rsplit('<｜Assistant｜>', 1)[-1]
        seg = seg.split('<｜Assistant｜>', 1)[0]
        return seg
    return s

def _canonicalize_key(k: str) -> str:
    s = _normalize_text(k)
    return s

def compute_score(data_source, solution_str, ground_truth, extra_info=None, hparams=None):
    if '|im_start|' in solution_str:
        solution_str = _assistant_segment(solution_str)
    elif '<｜Assistant｜>' in solution_str:
        solution_str = _distill_segment(solution_str)

    length_penalty = False
    if extra_info and 'length_penalty' in list(extra_info.keys()):
        length_penalty = True

    think_cnt = _count_blocks(solution_str, "think")
    answer_cnt = _count_blocks(solution_str, "answer")
    confidence_cnt = _count_blocks(solution_str, "confidence")

    format_bonus = 1.0
    if not (think_cnt == 1) or not (answer_cnt == 1) or not (confidence_cnt == 1):
        format_bonus = 0.0

    think = _last_block(solution_str, "think") or ""
    pred_answer_raw = _first_block(solution_str, "answer") or ""
    conf_raw = _first_block(solution_str, "confidence") or ""

    if not pred_answer_raw:
        maybe = _extract_answer_fallback(solution_str)
        if maybe is not None:
            pred_answer_raw = maybe

    pa_num = _try_float(pred_answer_raw)
    gt_num = _try_float(ground_truth)

    if pa_num is not None and gt_num is not None:
        pred_answer = pa_num
        gt_answer = gt_num
    else:
        pred_answer = _canonicalize_key(pred_answer_raw)
        gt_answer = _canonicalize_key(str(ground_truth))

    pa_num2 = _try_float(pred_answer)
    gt_num2 = _try_float(gt_answer)
    numeric_match = (pa_num2 is not None and gt_num2 is not None and abs(pa_num2 - gt_num2) <= 1e-8)

    exact_match = (pred_answer == gt_answer)
    is_correct = bool(exact_match or numeric_match)
    acc = 1.0 if is_correct else 0.0

    def _parse_confidence(text):
        m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
        if not m:
            return 0.0, True
        q = float(m.group(0))
        add_confidence_penalty = False
        if not (q <= 1.0):
            add_confidence_penalty = True
            q = q / 100.0
        if math.isnan(q) or math.isinf(q):
            return 0.0, True
        return max(0.0, min(1.0, q)), add_confidence_penalty

    q, add_confidence_penalty = _parse_confidence(conf_raw)
    confidence_pen = -0.2 if add_confidence_penalty else 0.0
    brier = (q - acc) ** 2
    base_reward = acc - brier

    return {
        "is_correct": is_correct,
        "acc": acc,
        "q": q,
        "brier": brier,
        "base_reward": base_reward,
        "format_bonus": format_bonus,
        "pred_answer_raw": pred_answer_raw,
        "conf_raw": conf_raw,
        "solution": solution_str,
    }

def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_NAME, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok

def load_model_from_subfolder():
    cfg = AutoConfig.from_pretrained(BASE_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        FT_REPO,
        subfolder=FT_SUBFOLDER,
        config=cfg,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return model, "full_from_subfolder"

def load_val(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    return data

def build_prompts(items: List[Dict[str, Any]]) -> List[str]:
    prompts = []
    for it in items:
        p = it["prompt"]
        if isinstance(p, list) and len(p) > 0 and "content" in p[0]:
            prompts.append(p[0]["content"])  
        else:
            raise ValueError("Unexpected prompt format.")
    return prompts

@torch.no_grad()
def generate_one(model, tok, prompt: str):
    enc = tok(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)
    attn = enc.get("attention_mask", None)
    if attn is not None:
        attn = attn.to(model.device)

    gen_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
    )
    gen_text = tok.decode(gen_ids[0][input_ids.shape[1]:], skip_special_tokens=False)
    print('GENERATED TEXT', gen_text)
    return gen_text

def get_digit_token_ids(tok: AutoTokenizer):
    cand_strs = ["1", " 1", "0", " 0"]
    ids = {}
    for s in cand_strs:
        enc = tok.encode(s, add_special_tokens=False)
        if len(enc) == 1:
            tid = enc[0]
            dec = tok.decode([tid])
            if dec == s:
                ids[s] = tid
    return ids  

@torch.no_grad()
def probe_next_token_probs_after_confidence(tok, model, full_output: str, prompt_text: str):
    tag = "<confidence>"
    idx = full_output.find(tag)
    if idx == -1:
        return {"p1": None, "p0": None, "note": "no_conf_tag_in_generation"}

    prefix_gen = full_output[: idx + len(tag)]
    probe_text = prompt_text + prefix_gen

    enc = tok(probe_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)
    attn = enc.get("attention_mask", None)
    if attn is not None:
        attn = attn.to(model.device)

    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits[:, -1, :]  
    probs = torch.softmax(logits, dim=-1)[0]

    ids = get_digit_token_ids(tok)
    p1 = 0.0
    if "1" in ids:
        p1 += float(probs[ids["1"]])
    if " 1" in ids:
        p1 += float(probs[ids[" 1"]])

    p0 = 0.0
    if "0" in ids:
        p0 += float(probs[ids["0"]])
    if " 0" in ids:
        p0 += float(probs[ids[" 0"]])

    if (("1" not in ids and " 1" not in ids) or ("0" not in ids and " 0" not in ids)):
        return {"p1": None, "p0": None, "note": "digit_tokens_missing"}

    return {"p1": p1, "p0": p0, "note": "ok"}

def main():
    torch.set_grad_enabled(False)

    data = load_val(VAL_PATH)
    prompts = build_prompts(data)

    tok = load_tokenizer()
    model, mode = load_model_from_subfolder()
    print(f"Loaded model mode: {mode}")

    total_acc = 0.0
    total_reward = 0.0
    total_format = 0.0

    with open(OUT_JSONL, "w") as out_f:
        for i, (item, prompt_text) in enumerate(zip(data, prompts)):
            gen = generate_one(model, tok, prompt_text)

            score = compute_score(
                data_source=item.get("data_source", None),
                solution_str=gen,
                ground_truth=item.get("reward_model", {}).get("ground_truth", None),
                extra_info=item.get("extra_info", None),
                hparams=None
            )

            probe = probe_next_token_probs_after_confidence(
                tok=tok, model=model, full_output=gen, prompt_text=prompt_text
            )

            row = {
                "index": item.get("extra_info", {}).get("index", i),
                "split": item.get("extra_info", {}).get("split", None),
                "is_correct": score["is_correct"],
                "acc": score["acc"],
                "q": score["q"],
                "brier": score["brier"],
                "base_reward": score["base_reward"],
                "format_bonus": score["format_bonus"], # quick check to make sure generation config is correct
                "pred_answer_raw": score["pred_answer_raw"],
                "conf_raw": score["conf_raw"],
                "solution": score["solution"],
                "p_conf_1": probe.get("p1", None),
                "p_conf_0": probe.get("p0", None),
                "probe_note": probe.get("note", None),
                "num_choices": item.get("extra_info", {}).get("num_choices", None),
                "gold_text": item.get("extra_info", {}).get("gold_text", None),
                "ground_truth": item.get("reward_model", {}).get("ground_truth", None),
                "original_question": item.get("extra_info", {}).get("original_question", None),
                "data_source": item.get("data_source", None),
            }
            out_f.write(json.dumps(row) + "\n")

            total_acc += score["acc"]
            total_reward += score["base_reward"]
            total_format += score["format_bonus"]

            if (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(data)}] acc={total_acc/(i+1):.4f} "
                      f"reward={total_reward/(i+1):.4f} format={total_format/(i+1):.4f}")

    n = len(data)
    print("\n==== Aggregates ====")
    print(f"Examples: {n}")
    print(f"Accuracy: {total_acc / n:.4f}")
    print(f"Base reward (acc - brier): {total_reward / n:.4f}")
    print(f"Format compliance rate: {total_format / n:.4f}")
    print(f"Per-example results saved to: {OUT_JSONL}")

if __name__ == "__main__":
    main()
