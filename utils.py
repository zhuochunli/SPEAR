import json
import re
from datasets import load_dataset, concatenate_datasets
import glob
import os
from math_verify import LatexExtractionConfig, ExprExtractionConfig, parse, verify
import random
from collections import defaultdict


def build_prompt(question, options=None, dataset='gsm8k'):
    if 'gsm8k' in dataset:
        return  question + "\nLet's think step by step."
    elif dataset == 'math' or dataset == 'math500':
        return question + "\nLet's think step by step and put the final answer in \\boxed{}."
    elif dataset == 'gpqa':
        random.shuffle(options)    # there is no A,B,C,D sequence in GPQA
        return question + " Put the final answer in \\boxed{}.\n" + '\n'.join(options) + "\n\nPlease make your think process as short as possible." 
    elif dataset == 'commonsenseQA':    # there is A,B,C,D sequence in commonsenseQA
        return question + " Put the final answer in \\boxed{}.\n" + '\n'.join(options)
    elif dataset == 'strategyQA':
        return question + "\nAnswer true or false, then explain your reasoning."
    elif dataset == 'alpaca2':
        return question
    
def split_sample(sample, dataset='gsm8k'):
    '''
    split question, answer and rationale for dataset GSM8K
    :param sample: dict{}
    :return: ques, ration, ans
    '''
    if dataset == 'gsm8k':
        ques = sample['question'].strip()
        ration = sample['answer'].strip()
        final_ans = sample['answer'].split('####')[1].strip()
    elif dataset == 'math':
        ques = sample['problem'].strip()
        ration = sample['solution'].strip()
        # math has no gold final answer
        final_ans = sample['solution'].strip()
    elif dataset == 'math500':
        ques = sample['problem'].strip()
        ration = sample['solution'].strip()
        final_ans = sample['solution'].strip()
    elif dataset == 'strategyQA':
        ques = sample['question'].strip()
        ration = sample['facts'].strip()
        final_ans = str(sample['answer']).strip().lower()
        ration = final_ans + '.\t' + ration  # rationale should also include the final answer
    elif dataset == 'commonsenseQA':
        ques = sample['question'].strip()
        options = [f"{l}) {t}" for l, t in zip(sample['choices']['label'], sample['choices']['text'])]
        final_ans = sample['answerKey'].strip()  # "A" instead of "ignore"
        return ques, options, final_ans
    elif dataset == 'logiQA':
        ques = sample['query'].strip()
        context = sample['context'].strip()
        options = sample['options']
        final_ans = str(sample['correct_option']).strip()
        return context, ques, options, final_ans
    elif dataset == 'gpqa':
        ques = sample['Question'].strip()
        options = [sample['Correct Answer'], sample['Incorrect Answer 1'], sample['Incorrect Answer 2'], sample['Incorrect Answer 3']]
        final_ans = str(sample['Correct Answer']).strip()
        return ques, options, final_ans
    elif dataset == 'alpaca2':
        ques = sample['instruction'].strip()
        ration = sample['output'].strip()
        final_ans = sample['output'].strip()

    return ques, ration, final_ans


def cleanup(pred, dataset='gsm8k', options=None):
    """
    :param pred: generated text
    :param dataset: task
    :return: [cleaned_text, final_prediction]

    options: only deal with logiQA dataset
    """
    if dataset == 'gsm8k' or dataset == 'svamp':
        pred = pred.strip()
        temp = pred

        struct_ans_flag = False
        for answer_prefix in ['\nAnswer', 'Therefore, the answer is']:
            if answer_prefix in pred:
                temp = pred.split(answer_prefix)[1].strip()
                struct_ans_flag = True
                break

        # extract all numbers in prediction
        temp_ori = [item for item in re.findall(r'-?\d+\.?\$?,?\d*', temp)]
        temp = [item.strip('.') for item in re.findall(r'-?\d+\.?\d*', temp.replace(',', ''))]

        if len(temp) == 0:
            final_pred = 'ABSOLUTE_WRONG_FINAL_ANS'
            if struct_ans_flag:
                answer_prefix_idx = pred.index(answer_prefix)
                next_word = pred[answer_prefix_idx + len(answer_prefix):].split()
                if next_word[0] == ':':
                    if len(next_word) == 1:
                        next_word = ' '
                    else:
                        next_word = ': ' + next_word[1]
                else:
                    next_word = ' ' + next_word[0]
                pred = pred[:answer_prefix_idx + len(answer_prefix)] + next_word

        elif struct_ans_flag:
            final_pred = temp[0]
            answer_prefix_idx = pred.index(answer_prefix)
            if final_pred in pred[answer_prefix_idx:]:
                temp_idx = pred[answer_prefix_idx:].index(final_pred)
                pred = pred[:answer_prefix_idx + temp_idx + len(final_pred)]
            else:
                next_word = pred[answer_prefix_idx + len(answer_prefix):].split()
                if next_word[0] == ':':
                    next_word = ': ' + next_word[1]
                else:
                    next_word = ' ' + next_word[0]
                pred = pred[:answer_prefix_idx + len(answer_prefix)] + next_word

        elif not struct_ans_flag:
            final_pred = temp[-1]  # the last number
            if final_pred in pred:
                pred = pred[:pred.index(final_pred) + len(final_pred)]
            elif temp_ori[-1] in pred:
                pred = pred[:pred.index(temp_ori[-1]) + len(temp_ori[-1])]
            else:
                pass
        else:
            raise RuntimeError()

    elif dataset == 'strategyQA':
        if len(pred) == 0:
            final_pred = 'ABSOLUTE_WRONG_FINAL_ANS'
        elif "true" in pred.lower():
            final_pred = 'true'
        elif "false" in pred.lower():
            final_pred = 'false'
        else:
            final_pred = 'ABSOLUTE_WRONG_FINAL_ANS'

    return pred, final_pred


def verify_preds(pred, gold_answer, dataset='gsm8k'):
    if 'gsm8k' in dataset:
        try:
            gold_parsed = parse(gold_answer, extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()],
                                fallback_mode="first_match", extraction_mode="any_match")
            answer_parsed = parse(pred, extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()],
                                  fallback_mode="first_match", extraction_mode="any_match")
            return verify(answer_parsed, gold_parsed)
        except Exception as e:
            # Catch SymPy/math_verify crashes (like the FiniteSet TypeError)
            return False
            
    elif 'math' in dataset:
        try:
            gold_parsed = parse(gold_answer, extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()],
                                fallback_mode="first_match", extraction_mode="any_match")
            answer_parsed = parse(pred, extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()],
                                  fallback_mode="first_match", extraction_mode="any_match")
            return verify(answer_parsed, gold_parsed)
        except Exception as e:
            # Catch SymPy/math_verify crashes
            return False
            
    # --- GPQA / CommonsenseQA: \boxed{} explicitly requested in prompt ---
    # Primary: extract from \boxed{} (handles one level of nested braces)
    # Fallback: last line only (avoids false matches in reasoning body)
    elif dataset == 'gpqa':
        match = re.search(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', pred)
        if match:
            return gold_answer.lower() in match.group(1).strip().lower()
        # Fallback: check last line only — avoids matching gold answer mid-reasoning
        last_line = pred.strip().split('\n')[-1].lower()
        return gold_answer.lower() in last_line
    
    elif dataset == 'commonsenseQA':
        match = re.search(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', pred)
        if match:
            return match.group(1).strip().upper() == gold_answer.upper()  # "A" == "A"
        last_line = pred.strip().split('\n')[-1].lower()
        return gold_answer.lower() in last_line
    # --- StrategyQA: prompt asks for "true or false" on first or last line ---
    elif dataset == 'strategyQA':
        first_line = pred.strip().split('\n')[0].lower()
        last_line = pred.strip().split('\n')[-1].lower()
        if 'true' in first_line or 'true' in last_line:
            final_pred = 'true'
        elif 'false' in first_line or 'false' in last_line:
            final_pred = 'false'
        else:
            return False
        return final_pred == gold_answer.lower()

    return False


def my_load_dataset(dataset='gsm8k'):
    train_dataset, test_dataset = [], []
    if dataset == 'gsm8k':
        train_dataset = load_dataset('gsm8k', 'main', split='train', cache_dir='local_dataset/')
        test_dataset = load_dataset('gsm8k', 'main', split='test', cache_dir='local_dataset/')
    elif dataset == 'math':
        train_dir = 'local_dataset/MATH/train'
        test_dir = 'local_dataset/MATH/test'
        train_files = glob.glob(os.path.join(train_dir, '**', '*.json'), recursive=True)
        for file in train_files:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                train_dataset.append(data)
        test_files = glob.glob(os.path.join(test_dir, '**', '*.json'), recursive=True)
        for file in test_files:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                test_dataset.append(data)
                
        # train_dataset = load_dataset("chiayewken/competition_math", split='train', cache_dir='local_dataset/')
        # test_dataset = load_dataset("chiayewken/competition_math", split='test', cache_dir='local_dataset/')
    elif dataset == 'math500':
        train_dataset = load_dataset("HuggingFaceH4/MATH-500", split='test', cache_dir='local_dataset/')
        test_dataset = load_dataset("HuggingFaceH4/MATH-500", split='test', cache_dir='local_dataset/')
    elif dataset == 'commonsenseQA':
        train_dataset = load_dataset("tau/commonsense_qa", split='train', cache_dir='local_dataset/')
        test_dataset = load_dataset("tau/commonsense_qa", split='validation', cache_dir='local_dataset/')   # test set has no gold answer
    elif dataset == 'strategyQA':
        train_dataset = load_dataset("ChilleD/StrategyQA", split='train', cache_dir='local_dataset/')
        test_dataset = load_dataset("ChilleD/StrategyQA", split='test', cache_dir='local_dataset/')
    elif dataset == 'gpqa':
        access_token = os.getenv("HF_TOKEN")
        gpqa_main = load_dataset("Idavidrein/gpqa", 'gpqa_main', split='train', cache_dir='local_dataset/', token=access_token)
        gpqa_extended = load_dataset("Idavidrein/gpqa", 'gpqa_extended', split='train', cache_dir='local_dataset/', token=access_token)
        train_dataset = concatenate_datasets([gpqa_main, gpqa_extended])
        test_dataset = load_dataset("Idavidrein/gpqa", 'gpqa_diamond',  split='train', cache_dir='local_dataset/', token=access_token)
    elif dataset == 'alpaca2':
        # dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval_gpt4_baseline", split='eval', cache_dir='local_dataset/')
        dataset = load_dataset("json", data_files={"eval": "local_dataset/alpaca_eval/alpaca_eval_gpt4_baseline.json"}, split="eval")
        train_dataset = dataset
        test_dataset = dataset
    return train_dataset, test_dataset
