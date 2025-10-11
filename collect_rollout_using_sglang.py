#!/usr/bin/env python3
import argparse
import os
import socket
import random
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from openai import OpenAI
import pandas as df_pd
from verl import DataProto  # noqa: F401
import transformers
from verl.utils.reward_score.miromind import compute_score
from tqdm import tqdm
import pickle


# -----------------------------
# LLM call (thread target)
# -----------------------------
def _query_one(i, messages, model_str, base_url, api_key,
               max_tokens, temperature=0.6, top_p=0.95):
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=6000)
    resp = client.chat.completions.create(
        model=model_str,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    response_text = resp.choices[0].message.content
    return i, response_text


# -----------------------------
# Per-index files & atomic ops
# -----------------------------
def _items_dir(out_dir):
    p = os.path.join(out_dir, "items")
    os.makedirs(p, exist_ok=True)
    return p

def _item_path(out_dir, idx):
    return os.path.join(_items_dir(out_dir), f"{idx:08d}.pkl")

def _atomic_save_dataproto(path, dataproto):
    tmp = path + ".tmp"
    dataproto.save_to_disk(tmp)
    os.replace(tmp, path)  # atomic on POSIX


# -----------------------------
# Lock helpers (no TTL/steal)
# -----------------------------
def _locks_dirs(out_dir):
    lock_dir = os.path.join(out_dir, "_locks")
    os.makedirs(lock_dir, exist_ok=True)
    return lock_dir

def _lock_path(out_dir, idx):
    return os.path.join(_locks_dirs(out_dir), f"{idx:08d}.lock")

def try_acquire_idx(out_dir, idx):
    """Acquire per-index lock only if not already done and not locked."""
    # If result already exists, skip
    if os.path.exists(_item_path(out_dir, idx)):
        return False
    lp = _lock_path(out_dir, idx)
    try:
        fd = os.open(lp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(f"host={socket.gethostname()} pid={os.getpid()} time={time.time()}\n")
        return True
    except FileExistsError:
        return False

def release_lock(out_dir, idx):
    lp = _lock_path(out_dir, idx)
    try:
        os.unlink(lp)
    except FileNotFoundError:
        pass


def count_done(out_dir):
    """Global done = number of per-index pickles present."""
    try:
        return sum(1 for n in os.listdir(_items_dir(out_dir)) if n.endswith(".pkl"))
    except FileNotFoundError:
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-str", required=True)
    parser.add_argument("--max-len", type=int, default=32768)
    parser.add_argument("--base-url", default="http://localhost:30000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--presence_penalty", default=0, type=float)
    parser.add_argument("--frequency_penalty", default=0, type=float)
    parser.add_argument("--idle-sleep", type=float, default=0.25,
                        help="Sleep (s) before rescanning when all items are locked/done elsewhere.")
    parser.add_argument("--data-path", default="data/aaamo_32aime.parquet",
                        help="Parquet with a 'prompt' (chat messages), 'data_source', and 'reward_model' column.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    _items_dir(args.out_dir)  # ensure exists
    _locks_dirs(args.out_dir)

    # Load data + tokenizer
    df = df_pd.read_parquet(args.data_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_str)

    total_rows = len(df)

    # Pre-skip indices already done
    already_done = set()
    for name in os.listdir(_items_dir(args.out_dir)):
        if name.endswith(".pkl"):
            try:
                already_done.add(int(name.split(".")[0]))
            except Exception:
                pass

    print("len(already_done)", len(already_done))
    if os.environ.get('DEBUG', False):
        raise

    indices = [i for i in range(total_rows) if i not in already_done]

    # Rolling state
    next_ptr = 0
    in_flight = {}           # fut -> idx
    data_sources = {}
    reward_models = {}

    def top_up(pool):
        nonlocal next_ptr
        slots = args.concurrency - len(in_flight)
        if slots <= 0 or len(indices) == 0:
            return 0
        submitted = 0
        scanned = 0
        N = len(indices)
        while submitted < slots and scanned < N:
            i = indices[next_ptr]
            next_ptr = (next_ptr + 1) % N
            scanned += 1

            # Skip if item exists or lock cannot be acquired
            if not try_acquire_idx(args.out_dir, i):
                continue

            row = df.iloc[i]
            data_sources[i] = row["data_source"]
            reward_models[i] = row["reward_model"]

            fut = pool.submit(
                _query_one, i, row["prompt"], args.model_str, args.base_url, args.api_key, args.max_len
            )
            in_flight[fut] = i
            submitted += 1
        return submitted

    processed_here = 0
    target_total = len(indices)

    if target_total == 0:
        print("Nothing to do (everything already has a per-index pickle).")
        return

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        pbar = tqdm(total=target_total, desc="Completed (this worker)", leave=True)

        while True:
            new_submits = top_up(pool)

            if in_flight:
                for fut in as_completed(list(in_flight.keys()), timeout=None):
                    i = in_flight.pop(fut)
                    try:
                        _, text = fut.result()

                        # Build single-sample tensors (no padding needed)
                        prompt_text = tokenizer.apply_chat_template(
                            df.iloc[i]["prompt"], tokenize=False, add_generation_prompt=True
                        )

                        prompts_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"].long()
                        responses_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].long()
                        input_ids = torch.cat([prompts_ids, responses_ids], dim=1)
                        attention_mask = torch.ones_like(input_ids)

                        # Reward
                        try:
                            gt = reward_models[i]["ground_truth"]
                            score = compute_score(text, gt)["score"]
                        except Exception as e:
                            print(f"[WARN] reward failed; idx={i} err={e}")
                            score = float("nan")

                        # Pack and atomically save a per-index pickle
                        batch_dict = {
                            "prompts": prompts_ids,
                            "responses": responses_ids,
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                        }
                        non_tensor_batch = {
                            "data_source": [data_sources[i]],
                            "reward_model": [reward_models[i]],
                            "reward": [score],
                            "index": [i],
                            "saved_at": [datetime.utcnow().isoformat() + "Z"],
                        }
                        dataproto = DataProto.from_dict(tensors=batch_dict, non_tensors=non_tensor_batch)

                        path = _item_path(args.out_dir, i)
                        _atomic_save_dataproto(path, dataproto)

                        processed_here += 1
                        pbar.update(1)

                    except Exception as e:
                        print(f"[WARN] idx={i} failed: {e}")
                        # On failure, just release the lock so another attempt can happen later
                    finally:
                        release_lock(args.out_dir, i)

                    # re-enter to top up again
                    break
                continue

            # If nothing in flight, check global done
            if count_done(args.out_dir) >= total_rows:
                break
            if new_submits == 0:
                time.sleep(args.idle_sleep)
                continue

        pbar.close()

    if processed_here == 0:
        print("No new items processed in this run.")
    else:
        print(f"Finished. This worker saved {processed_here} items.")


if __name__ == "__main__":
    main()

