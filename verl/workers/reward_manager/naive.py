# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
import os, json, tempfile

class NaiveRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        
    def __call__(self, data: DataProto, parallel_confidence=False):
        """We will expand this function gradually based on the available datasets"""
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

        # with open("/home/sp2583/rlvr3/triviaqa_alias_map.json") as f:
        #     TRIVIAQA_ALIAS_MAP = json.load(f)

        def _canonicalize_key(k: str) -> str:
            s = _normalize_text(k)
            return s
            # return TRIVIAQA_ALIAS_MAP.get(s, s)

        def _safe_load_json(path: str):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except FileNotFoundError:
                return {}
            except Exception:
                return {}

        def _safe_write_json(path: str, obj):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            dir_ = os.path.dirname(path) if os.path.dirname(path) else "."
            with tempfile.NamedTemporaryFile("w", delete=False, dir=dir_, encoding="utf-8") as tmp:
                json.dump(obj, tmp, ensure_ascii=False, indent=2)
                tmp_path = tmp.name
            os.replace(tmp_path, path)


        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']
        if parallel_confidence:
            num_rollouts = data.meta_info['num_rollouts']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        score_record = []
        components_record = []
        generations_to_log = {}
        if parallel_confidence:
            groups_start_idxs = range(0, len(data), num_rollouts)
            answer_records = {}
            grouped_idxs = {}
            for i in range(len(data)):
                data_item = data[i]
                extra_info = data_item.non_tensor_batch.get('extra_info', None)
                orig_q = extra_info['original_question']
                if orig_q not in grouped_idxs:
                    grouped_idxs[orig_q] = []
                grouped_idxs[orig_q].append(i)
            
            for key in grouped_idxs.keys():
                if len(grouped_idxs[key]) != num_rollouts:
                    print('KEY', key)
                    print(grouped_idxs[key])
                    raise RuntimeError('Wrong batch num')
                
            for key in grouped_idxs.keys():
                answer_record = {}
                first_data_item = data[grouped_idxs[key][0]]
                extra_info = first_data_item.non_tensor_batch.get('extra_info', None)
                orig_q = extra_info['original_question']
                for i in grouped_idxs[key]:
                    data_item = data[i]  # DataProtoItem

                    ei = (data_item.non_tensor_batch.get('extra_info') or {})
                    if ei.get('original_question') != orig_q:
                                    print('ORIGINAL', orig_q)
                                    print('ACTUAL', ei.get('original_question'))
                                    raise RuntimeError("Parallel-confidence group mismatch")

                    prompt_ids = data_item.batch['prompts']
                    prompt_length = prompt_ids.shape[-1]

                    valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                    valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                    response_ids = data_item.batch['responses']
                    valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                    valid_response_ids = response_ids[:valid_response_length]

                    sequences = torch.cat((valid_prompt_ids, valid_response_ids))
                    solution_str = self.tokenizer.decode(sequences)
                    if '|im_start|' in solution_str:
                        solution_str = _assistant_segment(solution_str)
                    elif '<｜Assistant｜>' in solution_str:
                        solution_str = _distill_segment(solution_str)
                    
                    if orig_q is not None:
                        generations_to_log.setdefault(orig_q, []).append(solution_str)
                    
                    sol_assistant = solution_str

                    pred_answer_raw = _first_block(sol_assistant, "answer") or ""

                    if not pred_answer_raw:
                        maybe = _extract_answer_fallback(sol_assistant)
                        if maybe is not None:
                            pred_answer_raw = maybe

                    pa_num = _try_float(pred_answer_raw)

                    if pa_num is not None: # MCQ
                        pred_answer = pa_num
                    else: # free response
                        pred_answer = _canonicalize_key(pred_answer_raw)
                    answer_record[pred_answer] = answer_record.get(pred_answer, 0) + 1
                answer_records[key] = answer_record



        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            orig_q_for_log = (extra_info or {}).get('original_question')
            if orig_q_for_log is not None:
                if '|im_start|' in sequences_str:
                    sequences_str_for_log = _assistant_segment(sequences_str)
                elif '<｜Assistant｜>' in sequences_str:
                    sequences_str_for_log = _distill_segment(sequences_str)
                else:
                    sequences_str_for_log = sequences_str
                generations_to_log.setdefault(orig_q_for_log, []).append(sequences_str_for_log)

            if parallel_confidence:
                for key in grouped_idxs.keys():
                    if i in grouped_idxs[key]:
                        correct_key = key
                        break
                extra_info['parallel_confidence'] = True
                extra_info['answer_record'] = answer_records[correct_key]

            score, components = self.compute_score(
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

            record = {
                "sequences_str": sequences_str,
                "ground_truth": ground_truth,
                "index": extra_info["index"] if extra_info else None,
                "score": score
            }
            score_record.append(record)
            components_record.append(components)

        # generations_log_path = '/home/sp2583/rlvr/plausibleqa_generations.json'
        # if generations_to_log:
        #     existing = _safe_load_json(generations_log_path)
        #     if not isinstance(existing, dict):
        #         existing = {}
        #     for q, gen_list in generations_to_log.items():
        #         if q is None:
        #             continue
        #         bucket = existing.setdefault(q, [])
        #         bucket.extend(gen_list)
        #         seen = set()
        #         deduped = []
        #         for g in bucket:
        #             if g not in seen:
        #                 seen.add(g)
        #                 deduped.append(g)
        #         existing[q] = deduped
        #     _safe_write_json(generations_log_path, existing)

        return reward_tensor, score_record, components_record
