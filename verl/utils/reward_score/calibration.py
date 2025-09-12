from transformers import AutoTokenizer
import re
from typing import Dict, Optional
from math import isfinite
import unicodedata
from difflib import SequenceMatcher
from transformers import AutoTokenizer
from datasets import load_dataset
import json

import re, unicodedata
from datasets import load_dataset

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

    # m = re.fullmatch(r'.*?(-?\d+(?:\.\d+)?)\s*(?:years|yrs|yo|years old)?', s)
    # if m:
    #     return m.group(1)

    # s = re.sub(r'\bby\b.*$', '', s, flags=re.IGNORECASE)

    # s = re.sub(r'\s*[\(\[\{][^)\]\}]*[\)\]\}]\s*$', '', s)

    # s = unicodedata.normalize("NFKD", s)
    # s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # s = re.sub(r"[^a-z0-9\s]", " ", s)

    # s = re.sub(r'\s+', ' ', s).strip()

    # if s:
    #     parts = s.split()
    #     if parts[0] in _ARTICLES and len(parts) > 1:
    #         s = " ".join(parts[1:])

    # if s in _ABBR:
    #     s = _ABBR[s]


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

# try:
#     with open("/home/sp2583/rlvr3/triviaqa_alias_map.json") as f:
#         TRIVIAQA_ALIAS_MAP = json.load(f)
# except:
#     TRIVIAQA_ALIAS_MAP = None

def _canonicalize_key(k: str) -> str:
    s = _normalize_text(k)
    return s
    # if TRIVIAQA_ALIAS_MAP:
    #     return TRIVIAQA_ALIAS_MAP.get(s, s)
    # return 


def compute_score(data_source, solution_str, ground_truth, extra_info=None, hparams=None):


    if '|im_start|' in solution_str:
        solution_str = _assistant_segment(solution_str)
    elif '<｜Assistant｜>' in solution_str:
        solution_str = _distill_segment(solution_str)
    
    length_penalty = False
    if 'length_penalty' in list(extra_info.keys()):
        length_penalty = True
    
    parallel_confidence = extra_info.get('parallel_confidence', None)
    # print('PARALLEL_CONFIDENCE', parallel_confidence)
    
    
    # predict_distribution = False
    # try:
    #     elem1, elem2 = ground_truth
    #     percent_correct = elem1
    #     ground_truth = elem2
    #     predict_distribution = True
    # except:
    #     predict_distribution = False


    # ground_truth = ground_truth[ground_truth.index('<answer>') + len('<answer>'):]

    sol_assistant = solution_str
    print('SOLUTION', solution_str)

    think_cnt = _count_blocks(sol_assistant, "think")
    answer_cnt = _count_blocks(sol_assistant, "answer")
    confidence_cnt = _count_blocks(sol_assistant, "confidence")

    print(think_cnt, answer_cnt, confidence_cnt)

    format_bonus = 1.0
    if not (think_cnt == 1) or not (answer_cnt == 1) or not (confidence_cnt == 1):
        format_bonus = 0.0

    # if not (think_cnt == 1) or not (answer_cnt == 1):
    #     format_bonus = 0.0

    length = None
    if length_penalty and format_bonus == 1.0:
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
        thinking_trace_end = solution_str.index('</think>')
        length = len(tokenizer(solution_str[:thinking_trace_end], return_tensors='pt')['input_ids'][0])

    think = _last_block(sol_assistant, "think") or ""
    pred_answer_raw = _first_block(sol_assistant, "answer") or ""
    conf_raw = _first_block(sol_assistant, "confidence") or ""

    if not pred_answer_raw:
        maybe = _extract_answer_fallback(sol_assistant)
        if maybe is not None:
            pred_answer_raw = maybe

    pa_num = _try_float(pred_answer_raw)
    gt_num = _try_float(ground_truth)

    if pa_num is not None and gt_num is not None: # plausible
        pred_answer = pa_num
        gt_answer = gt_num
    else: # hotpot/trivia/etc.
        pred_answer = _canonicalize_key(pred_answer_raw)
        gt_answer = _canonicalize_key(str(ground_truth))

    print('PRED ANSWER', pred_answer)
    print('GT', gt_answer)
    # if predict_distribution:
    #     print('PERCENT CORRECT', percent_correct)

    pa_num2 = _try_float(pred_answer)
    gt_num2 = _try_float(gt_answer)
    numeric_match = (pa_num2 is not None and gt_num2 is not None and abs(pa_num2 - gt_num2) <= 1e-8)

    exact_match = (pred_answer == gt_answer)
    is_correct = bool(exact_match or numeric_match)

    def _parse_confidence(text):
        import re, math
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
    confidence_pen = 0.0
    if add_confidence_penalty:
        confidence_pen = - 0.2
    print('CONFIDENCE', q)
    print('FORMAT BONUS', format_bonus)
    acc = 1.0 if is_correct else 0.0

    # if not predict_distribution:

    brier = (q - acc) ** 2
    base_reward = acc - brier

    if not length_penalty and not parallel_confidence:
        print('NORMAL')

        w_acc = 0.35
        w_base = 0.35
        w_format = 0.3

        score =  w_acc * acc + w_base * base_reward + w_format * format_bonus + confidence_pen
        score = max(-1.0, min(1.0, score))
    
    elif not parallel_confidence:
        print('LENGTH PENALTY')
        if length is None:
            length_reward = - 1.0
        w_acc = 0.35
        w_base = 0.35
        w_format = 0.3
        w_length = 0.0

        score =  w_acc * acc + w_base * base_reward + w_format * format_bonus + confidence_pen
        score = max(-1.0, min(1.0, score))
    
    else:
        w_acc = 0.0
        w_base = 1.0
        w_format = 1.0
        answer_record = extra_info['answer_record']
        num_sampled = answer_record.get(pred_answer, 0)
        consistency = num_sampled / sum(list(answer_record.values()))
        import math
        w_base = 1 / (consistency ** 0.25)
        print('CONSISTENCY', consistency)
        base_reward = - (q - consistency) ** 2
        print('CONSISTENCY REWARD', base_reward)

        score =  w_acc * acc + w_base * base_reward + w_format * format_bonus + confidence_pen
        score = max(-1.0, min(1.0, score))


        
    # else:
    #     brier = (percent_correct - q) ** 2
    #     base_reward = 1 - brier 
    #     score = 0.4 * acc + 0.4 * base_reward + 0.2 * format_bonus + confidence_pen 
    #     score = max(-1.0, min(1.0, score))

    # score = 0.7 * base_reward + 0.3 * format_bonus + confidence_pen
    # score = max(-1.0, min(1.0, score))
    
    print('SCORE', score)

    components = {
        # "exact_match": exact_match,
        "verbalized confidence": q,
        "acc": acc,
        "1 - brier": 1 - brier,
        "format_bonus": format_bonus,
        "total score": score,
        "confidence_pen": confidence_pen,
        "think_cnt": think_cnt,
        "answer_cnt": answer_cnt,
        "confidence_cnt": confidence_cnt,
    }
    if parallel_confidence:
        components['consistency'] = consistency
        components['consistency_brier'] = 1 - abs(q - consistency)

    return score, components


