import json
import torch
import os
import re
import numpy as np
import random
from datasets import load_dataset, DatasetDict, Dataset
import argparse
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from all_prompts import think_prompt
# from unsloth.chat_templates import get_chat_template
from math_verify import LatexExtractionConfig, ExprExtractionConfig, parse, verify
from utils import build_prompt, split_sample


# Load dataset from JSON file
def load_json_dataset(jsonl_path):
    # jsonl data: one json file each line
    with open(jsonl_path, "r") as f:
        data = [json.loads(line) for line in f]
    formatted_data = [{
        "input": item["prompt"].strip(),
        # "output": item["answer"].strip()
        # "output": item["deepseek_answer"].strip()
        "output": "<think> "+item["deepseek_rationale"].strip()+" </think> "+"<answer> "+item["deepseek_answer"].strip()+" </answer>"
    } for item in data]
    return Dataset.from_list(formatted_data)


def preprocess_dataset(json_path, seed):
    dataset = load_json_dataset(json_path)
    # Shuffle and split (90% train, 10% validation)
    shuffled_data = dataset.shuffle(seed=seed)
    split_idx = int(0.9 * len(shuffled_data))

    dataset_dict = DatasetDict({
        "train": Dataset.from_dict(shuffled_data[:split_idx]),
        "validation": Dataset.from_dict(shuffled_data[split_idx:])
    })
    return dataset_dict


def build_formatting_func(tokenizer):
    def formatting_func(example):
        messages = [
            # {"role": "system", "content": "You are a helpful assistant."},
            # {"role": "system", "content": think_prompt},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
        texts = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = False)
        return { "text" : texts, }
    return formatting_func


# def rl_preprocess_dataset(jsonl_path, dataset, seed):
#     # jsonl data: one json file each line
#     with open(jsonl_path, "r") as f:
#         data = [json.loads(line) for line in f]
#     formatted_data = [{
#         "input": item["prompt"].strip(),
#         # "solution": item["answer"].strip(),
#         "solution": split_sample(item, dataset),
#         "deepseek_rationale": item["deepseek_rationale"].strip(),
#         "deepseek_answer": item["deepseek_answer"].strip(),
#     } for i,item in enumerate(data)]
#     dataset =  Dataset.from_list(formatted_data)
#     shuffled_data = dataset.shuffle(seed=seed)
#     return shuffled_data

def rl_preprocess_dataset(jsonl_path, task_type, seed): # renamed 'dataset' to 'task_type' to avoid confusion
    with open(jsonl_path, "r") as f:
        data = [json.loads(line) for line in f]
    
    formatted_data = []
    for i, item in enumerate(data):
        # Ensure split_sample returns a string
        solution = split_sample(item, task_type)[-1]
        if isinstance(solution, list):
            solution = " ".join(map(str, solution)) # Force list to string
            
        formatted_data.append({
            "input": str(item.get("prompt", "")).strip(),
            "solution": str(solution).strip(),
            "deepseek_rationale": str(item.get("deepseek_rationale", "")).strip(),
            "deepseek_answer": str(item.get("deepseek_answer", "")).strip(),
        })

    dataset = Dataset.from_list(formatted_data)
    shuffled_data = dataset.shuffle(seed=seed)
    return shuffled_data

def rl_make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": think_prompt},
            {"role": "user", "content": example["input"]},
        ],
    }

def tokenize_function(example, tokenizer, max_length=2048):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        # {"role": "system", "content": think_prompt},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    full_text = tokenizer.apply_chat_template(messages,
                    add_generation_prompt=False,
                    tokenize=False)
    prompt_text = tokenizer.apply_chat_template(messages[:-1],
                    add_generation_prompt=False,
                    tokenize=False)

    # 1) Tokenize prompt WITHOUT padding → get real length
    prompt_batch = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )
    prompt_len = prompt_batch.input_ids.shape[-1]

    # 2) Tokenize full sequence WITH padding/truncation
    batch = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids      = batch.input_ids[0]
    attention_mask = batch.attention_mask[0]

    # 3) Build labels: mask only prompt tokens and pads
    labels = input_ids.clone()
    labels[:prompt_len] = -100
    labels[attention_mask == 0] = -100    # mask padding only, not EOS
    # labels[input_ids == tokenizer.pad_token_id] = -100

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }