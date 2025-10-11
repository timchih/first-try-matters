#!/usr/bin/env python3
import argparse
import errno
import gc
import json
import os
import pickle
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)

import ipdb
import numpy as np
import pandas as pd
import torch
import transformers
from matplotlib import pyplot as plt
from openai import OpenAI
from system_prompts import SYSTEM_PROMPT_1, SYSTEM_PROMPT_2, SYSTEM_PROMPT_3
from tqdm import tqdm
from transformers import AutoTokenizer

from verl.utils.reward_score.miromind import compute_score
from verl.utils.torch_functional import get_response_mask

# ---- Argument parsing (now also accepts run-id and system-prompt-id) ----
parser = argparse.ArgumentParser(description="Run rollout post-processing")
parser.add_argument('--debug', action='store_true',
                    help="Run in debug mode: one example + ipdb break")
parser.add_argument('--run-identifier', type=str, default="run_64k_prompt1",
                    help="Directory tag inserted between save_dir's parent and basename")
parser.add_argument('--system-prompt-id', type=int, choices=[1, 2, 3], default=1,
                    help="Which SYSTEM_PROMPT to use: 1, 2, or 3")
args = parser.parse_args()

DEBUG = args.debug
RUN_IDENTIFIER = args.run_identifier
_PROMPTS = {1: SYSTEM_PROMPT_1, 2: SYSTEM_PROMPT_2, 3: SYSTEM_PROMPT_3}
SYSTEM_PROMPT = _PROMPTS[args.system_prompt_id]

print(f"[CONFIG] run_identifier={RUN_IDENTIFIER}  system_prompt_id={args.system_prompt_id}")

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="EMPTY",
    timeout=1200
)

def parse_response(response):
    try:
        extracted = eval(response[response.rfind('['):response.rfind(']')+1])
        extracted = [(int(a), b) for a, b in extracted]
        for pos, cand in extracted:
            assert isinstance(pos, int)
            assert isinstance(cand, str)
        return extracted
    except Exception as e:
        return []

def magistral_text_processor(item, tokenizer):
    if DEBUG:
        ipdb.set_trace()
    text = tokenizer.decode(
        item.batch["input_ids"][item.batch["attention_mask"].bool()]
    )
    lines = text.split('\n')
    lineno = 2

    buffer = []
    for line in lines[7:]:
        buffer.append(line)
        if "[/INST]<think>" in line:
            break
    lines = lines[:7] + ['\n'.join(buffer)] + lines[7+len(buffer):]

    def process_first_line(x):
        for special_tok in ["Here, provide a concise summary that reflects your reasoning and presents a clear final answer to the user.[/SYSTEM_PROMPT][INST]", "[/INST]<think>"]:
            x = x.replace(special_tok, '')
        return x
    processed_lines = ["Problem statement:", process_first_line(lines[7]), "Model solution:"+"\n1: <think>\n"]

    for line in lines[8:]:
        processed_lines.append(f'{lineno}: {line}')
        lineno += 1
        if '</think>' in line:
            break

    return '\n'.join(processed_lines)

def qwen3_text_processor(item, tokenizer):
    if DEBUG:
        ipdb.set_trace()
    text = tokenizer.decode(
        item.batch["input_ids"][item.batch["attention_mask"].bool()]
    )
    lines = text.split('\n')
    lineno = 1
    processed_lines = []
    start_numbering = False

    for line in lines:
        if line == '<think>':
            start_numbering = True
        if line.strip() == '</think>':
            break
        for special_tok in ['<|im_start|>', '<|im_end|>']:
            line = line.replace(special_tok, '')
        if line == 'user':
            line = 'Problem statement:'
        if line == 'assistant':
            line = 'Model solution:'
        if start_numbering:
            processed_lines.append(f'{lineno}: {line}')
            lineno += 1
        else:
            processed_lines.append(line)

    return '\n'.join(processed_lines)

def qwen25_text_processor(item, tokenizer):
    if DEBUG:
        ipdb.set_trace()
    text = tokenizer.decode(
        item.batch["input_ids"][item.batch["attention_mask"].bool()]
    )
    lines = text.split('\n')
    problem_lines = lines[lines.index('<|im_start|>user')+1:lines.index('<|im_start|>assistant')]
    problem_lines[-1] = problem_lines[-1].replace('<|im_end|>', '')
    processed_lines = ["Problem statement:", "\n".join(problem_lines), "Model solution:"]

    lineno = 1
    for i, line in enumerate(lines[lines.index('<|im_start|>assistant')+1:]):
        processed_lines.append(f'{lineno}: {line}')
        lineno += 1
        if '</think>' in line:
            break

    return '\n'.join(processed_lines)

def mimo7b_text_processor(item, tokenizer):
    if DEBUG:
        ipdb.set_trace()
    text = tokenizer.decode(
        item.batch["input_ids"][item.batch["attention_mask"].bool()]
    )
    lines = text.split('\n')
    problem_lines = lines[lines.index('<|im_start|>user')+1:lines.index('<|im_start|>assistant')]
    problem_lines[-1] = problem_lines[-1].replace('<|im_end|>', '')
    processed_lines = ["Problem statement:", "\n".join(problem_lines), "Model solution:"]

    lineno = 1
    for i, line in enumerate(lines[lines.index('<|im_start|>assistant')+1:]):
        processed_lines.append(f'{lineno}: {line}')
        lineno += 1
        if '</think>' in line:
            break

    return '\n'.join(processed_lines)

def r1_distill_llama_8b_text_processor(item, tokenizer):
    if DEBUG:
        ipdb.set_trace()
    text = tokenizer.decode(
        item.batch["input_ids"][item.batch["attention_mask"].bool()]
    )
    lines = text.split('\n')

    buffer = []
    for line in lines:
        buffer.append(line)
        if "<｜Assistant｜><think>" in line:
            break
    lines = ['\n'.join(buffer)] + lines[len(buffer):]
    lineno = 2
    def process_first_line(x):
        for special_tok in ["<｜Assistant｜><think>", "<｜Assistant｜>", "<｜User｜>", "<｜end▁of▁sentence｜>", "<｜begin▁of▁sentence｜>"]:
            x = x.replace(special_tok, '')
        return x
    processed_lines = ["Problem statement:", process_first_line(lines[0]), "Model solution:", "1: <think>"]

    for line in lines[1:]:
        processed_lines.append(f'{lineno}: {line}')
        lineno += 1
        if '</think>' in line:
            break

    return '\n'.join(processed_lines)

def llama_8b_text_processor(item, tokenizer):
    if DEBUG:
        ipdb.set_trace()
    text = tokenizer.decode(
        item.batch["input_ids"][item.batch["attention_mask"].bool()]
    )
    lines = text.split('\n')

    problem_statement_last_line = 0
    for i, line in enumerate(lines):
        if line.endswith('<|eot_id|><|start_header_id|>assistant<|end_header_id|>'):
            problem_statement_last_line = i
            break
    problem_statement = text[text.find('<|eot_id|><|start_header_id|>user<|end_header_id|>')+len('<|eot_id|><|start_header_id|>user<|end_header_id|>'):text.find('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')]
    processed_lines = ["Problem statement:", problem_statement, "Model solution:"]

    lineno = 1
    for line in lines[problem_statement_last_line+1:]:
        processed_lines.append(f'{lineno}: {line}')
        lineno += 1
        if '</think>' in line:
            break

    return '\n'.join(processed_lines)

def mimo_text_processor(item, tokenizer):
    if DEBUG:
        ipdb.set_trace()
    text = tokenizer.decode(
        item.batch["input_ids"][item.batch["attention_mask"].bool()]
    )
    lines = text.split('\n')

    buffer = []
    for line in lines[3:]:
        if line != '<|im_start|>assistant':
            buffer.append(line)
        else:
            break
    lines = lines[:3] + ['\n'.join(buffer)] + lines[3+len(buffer):]
    lineno = 1
    def process_first_line(x):
        for special_tok in ["<|im_end|>"]:
            x = x.replace(special_tok, '')
        return x
    processed_lines = ["Problem statement:", process_first_line(lines[3]), "Model solution:"]

    for line in lines[5:]:
        processed_lines.append(f'{lineno}: {line}')
        lineno += 1
        if '</think>' in line:
            break

    return '\n'.join(processed_lines)

def gpt_text_processor(item, tokenizer):
    """
    import torch
    import pickle
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained('openai/gpt-oss-20b')
    batch = pickle.load(open('rollouts/gpt-oss-20b-v2/batches/val_batch_0.pkl', 'rb'))
    input_ids = torch.split(batch.batch['input_ids'][batch.batch['attention_mask'].bool()], batch.batch['attention_mask'].sum(dim=1).tolist())
    texts = tokenizer.batch_decode(input_ids)
    thinking_texts = [t[t.rfind('assistantanalysis')+len('assistantanalysis'):t[0].rfind('assistantfinal')] for t in texts]
    answer_texts = [t[t.rfind('assistantfinal')+len('assistantfinal'):] for t in texts]
    tokenized_thinking = tokenizer(thinking_texts)
    tokenized_answer = tokenizer(answer_texts)
    sum(len(l) for l in tokenized_thinking.input_ids) / len(tokenized_thinking.input_ids)  # 5584
    sum(len(l) for l in tokenized_answer.input_ids) / len(tokenized_answer.input_ids)  # 1910
    """
    text = tokenizer.decode(
        item.batch["input_ids"][item.batch["attention_mask"].bool()]
    )
    def process_first_line(x):
        for special_tok in ["<|im_end|>"]:
            x = x.replace(special_tok, '')
        return x
    processed_lines = ["Problem statement:", text[text.find('<|start|>user<|message|>')+len('<|start|>user<|message|>'):text.find('<|end|><|start|>assistant')], "Model solution:"]
    thinking_part = text[text.rfind('assistantanalysis')+len('assistantanalysis'):text.rfind('assistantfinal')]
    lines = thinking_part.split("\n")
    lineno = 1

    for line in lines[5:]:
        processed_lines.append(f'{lineno}: {line}')
        lineno += 1
        if '</think>' in line:
            break

    return '\n'.join(processed_lines)

def r1_0528_text_processor(item, tokenizer):
    """
    import torch
    import pickle
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained('openai/gpt-oss-20b')
    batch = pickle.load(open('rollouts/gpt-oss-20b-v2/batches/val_batch_0.pkl', 'rb'))
    input_ids = torch.split(batch.batch['input_ids'][batch.batch['attention_mask'].bool()], batch.batch['attention_mask'].sum(dim=1).tolist())
    texts = tokenizer.batch_decode(input_ids)
    thinking_texts = [t[t.rfind('assistantanalysis')+len('assistantanalysis'):t[0].rfind('assistantfinal')] for t in texts]
    answer_texts = [t[t.rfind('assistantfinal')+len('assistantfinal'):] for t in texts]
    tokenized_thinking = tokenizer(thinking_texts)
    tokenized_answer = tokenizer(answer_texts)
    sum(len(l) for l in tokenized_thinking.input_ids) / len(tokenized_thinking.input_ids)  # 5584
    sum(len(l) for l in tokenized_answer.input_ids) / len(tokenized_answer.input_ids)  # 1910
    """
    if DEBUG:
        ipdb.set_trace()
    text = tokenizer.decode(
        item.batch["input_ids"][item.batch["attention_mask"].bool()]
    )
    def process_first_line(x):
        for special_tok in ["<|im_end|>"]:
            x = x.replace(special_tok, '')
        return x
    processed_lines = ["Problem statement:", text[text.find('<|start|>user<|message|>')+len('<|start|>user<|message|>'):text.find('<|end|><|start|>assistant')], "Model solution:"]
    thinking_part = text[text.rfind('assistantanalysis')+len('assistantanalysis'):text.rfind('assistantfinal')]
    lines = thinking_part.split("\n")
    lineno = 1

    for line in lines[5:]:
        processed_lines.append(f'{lineno}: {line}')
        lineno += 1
        if '</think>' in line:
            break

    return '\n'.join(processed_lines)

def process_one(messages, save_path, raw_text):
    lock_path = save_path + '.lock'
    try:
        if not os.path.exists(save_path):
            resp = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages,
                max_tokens=32768,
                temperature=1.0,
                top_p=1.0,
                # extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
            item = {
                "messages": messages,
                "response": resp.choices[0].message.content,
                "reasoning_content": resp.choices[0].message.reasoning_content,
                "ground_truth": messages[-1]["ground_truth"],
                "data_source": messages[-1]["data_source"],
                "reward_score": str(messages[-1]["reward_score"]),
                "raw_text": raw_text,
            }
            candidates = parse_response(item['response'])
            candidate_correctness_1 = [compute_score(candidate, item['ground_truth'])['acc'] for lineno, candidate in candidates]
            candidate_correctness_2 = [compute_score("\\boxed{"+candidate+"}", item['ground_truth'])['acc'] for lineno, candidate in candidates]
            candidate_correctness = [c1 or c2 for c1, c2 in zip(candidate_correctness_1, candidate_correctness_2)]
            candidate_answers = [candidate for lineno, candidate in candidates]
            candidate_position = [lineno for lineno, candidate in candidates]
            def find_total_line(text):
                _lines = text.split('\n')
                _linenos = []
                for _line in _lines:
                    try:
                        _linenos.append(int(_line[:_line.index(':')]))
                    except:
                        pass
                return max(_linenos)
            total_lines = find_total_line(item['messages'][1]['content'])
            item['candidate_correctness'] = candidate_correctness
            item['candidates'] = candidate_answers
            item['candidate_position'] = candidate_position
            item['total_lines'] = total_lines
            with open(save_path, "w") as out:
                json.dump(item, out, indent=4)
        else:
            item = json.load(open(save_path))
            if sorted(tuple(item.keys())) == sorted(("messages", "response", "reasoning_content", "ground_truth", "data_source", "reward_score", "raw_text")):
                # legacy json, no candidate correctness
                candidates = parse_response(item['response'])
                candidate_correctness_1 = [compute_score(candidate, item['ground_truth'])['acc'] for lineno, candidate in candidates]
                candidate_correctness_2 = [compute_score("\\boxed{"+candidate+"}", item['ground_truth'])['acc'] for lineno, candidate in candidates]
                candidate_correctness = [c1 or c2 for c1, c2 in zip(candidate_correctness_1, candidate_correctness_2)]
                candidate_answers = [candidate for lineno, candidate in candidates]
                candidate_position = [lineno for lineno, candidate in candidates]
                total_lines = int(item['messages'][1]['content'].split('\n')[-5].split(':')[0])
                item['candidate_correctness'] = candidate_correctness
                item['candidates'] = candidate_answers
                item['candidate_position'] = candidate_position
                item['total_lines'] = total_lines
                with open(save_path, "w") as out:
                    json.dump(item, out, indent=4)
            else:
                # newer json, but not guaranteed everything's format is up to date
                assert sorted(tuple(item.keys())) == sorted(("messages", "response", "reasoning_content", "ground_truth", "data_source", "reward_score", "raw_text", "candidate_correctness", "candidates", "candidate_position", "total_lines"))
                assert isinstance(item['candidate_position'], list) and (len(item['candidate_position']) == 0 or isinstance(item['candidate_position'][0], int))
                

        # Debug breakpoint before writing out the file
        if DEBUG:
            print(f"[DEBUG] Entering ipdb for {save_path}")
            ipdb.set_trace()

    except Exception as e:
        print(f"[ERROR] Failed to process {save_path}: {e}")
        traceback.print_exc()

        tb = e.__traceback__
        while tb.tb_next:  # go to the last frame
            tb = tb.tb_next
        print(f"Error at {tb.tb_frame.f_code.co_filename}:{tb.tb_lineno}")
    finally:
        # release lock so crashed jobs can be retried
        try:
            os.remove(lock_path)
        except OSError:
            pass

packages = [
    {
        "root_dir": "rollouts/qwen3-8b-64k/items",
        "save_dir": 'candidate_positions/qwen3-8b-aaamo-gpt-oss-120b',
        "tokenizer": AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"),
        "processor": qwen3_text_processor
    }
    # ...
]

for package in packages:
    root_dir = package["root_dir"]
    save_dir = package["save_dir"]
    tokenizer = package["tokenizer"]
    processor = package["processor"]

    print(root_dir)

    BATCH_SIZE = 1 if DEBUG else 16
    # executor   = ThreadPoolExecutor(max_workers=BATCH_SIZE)
    executor   = ProcessPoolExecutor(max_workers=BATCH_SIZE)
    inflight   = set()

    # ---- Main loop ----
    for k, fn in enumerate(sorted(os.listdir(root_dir))):
        if not fn.endswith(".pkl"):
            continue
        if DEBUG and k > 0:
            break  # only one batch in debug

        batch = pickle.load(open(os.path.join(root_dir, fn), "rb"))
        try: response_mask_1 = get_response_mask(batch.batch['responses'], tokenizer.eos_token_id)
        except: response_mask_1 = get_response_mask(batch.batch['responses'], tokenizer.pad_token_id)
        try: response_mask_2 = get_response_mask(batch.batch['responses'], tokenizer.pad_token_id)
        except: response_mask_2 = get_response_mask(batch.batch['responses'], tokenizer.eos_token_id)
        response_mask = response_mask_1 & response_mask_2
        batch.batch['attention_mask'][:, -response_mask.shape[1]:] = response_mask

        for j, item in tqdm(enumerate(batch)):
            if DEBUG and j > 0:
                break  # only one item in debug

            text_block = processor(item, tokenizer)
            raw_text = tokenizer.decode(item.batch["input_ids"][item.batch["attention_mask"].bool()])
            if DEBUG:
                print(root_dir)
                print(text_block)
                import ipdb; ipdb.set_trace()
                break

            prompt = f"""
Analyze the following problem and its model solution.

{'-' * 40}
Below is the problem statement **followed by** the line-numbered model solution:
{'-' * 40}

{text_block}

{'-' * 40}
Perform necessary reasoning, understand what the problem asks for, and what the solutions is trying to do, then output the candidate answer in the format as specified.
"""

            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {"role": "user", "content": prompt},
            ]
            messages[-1]["ground_truth"] = item.non_tensor_batch["reward_model"]["ground_truth"]
            messages[-1]["data_source"] = item.non_tensor_batch["data_source"]
            messages[-1]["reward_score"] = str(item.non_tensor_batch["reward"])
            if DEBUG:
                ipdb.set_trace()

            path_1, path_2 = os.path.split(save_dir)
            os.makedirs(os.path.join(path_1, RUN_IDENTIFIER, path_2), exist_ok=True)
            save_path = os.path.join(path_1, RUN_IDENTIFIER, path_2, f"{fn[:-4]}_{j}.json")
            lock_path = save_path + '.lock'

            # skip if already done or claimed
            if os.path.exists(save_path) or os.path.exists(lock_path):
                continue

            # atomically claim
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
            except OSError as e:
                if e.errno in (errno.EEXIST, errno.EACCES):
                    continue
                else:
                    raise

            if DEBUG:
                # In debug mode: run inline so ipdb.set_trace() is in the main thread
                process_one(messages, save_path, raw_text)
                sys.exit(0)
            else:
                # submit job to the executor as normal
                fut = executor.submit(process_one, messages, save_path, raw_text)
                inflight.add(fut)

            # throttle to batch size
            if not DEBUG and len(inflight) >= BATCH_SIZE:
                done = next(as_completed(inflight))
                inflight.remove(done)
                done.result()

    # wait for remaining (only in normal mode)
    if not DEBUG:
        for fut in as_completed(inflight):
            fut.result()
        executor.shutdown()