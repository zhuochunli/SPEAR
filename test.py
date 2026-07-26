import json
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
import argparse
from datasets import load_dataset
from tqdm import tqdm
from utils import split_sample, my_load_dataset, build_prompt, verify_preds
from all_prompts import *
from math_verify import LatexExtractionConfig, ExprExtractionConfig, parse, verify


parser = argparse.ArgumentParser(description='Inference student model',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--model_path', '-m', default='meta-llama/Meta-Llama-3-8B-Instruct',
                    help='Base model path')
parser.add_argument('--lora_path', default=None, help='LoRA adapter path')
parser.add_argument('--dataset', default='gsm8k', choices=['gsm8k', 'math500', 'gpqa', 'commonsenseQA', 'strategyQA'],
                    help="Dataset for evaluation")
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size for inference')
parser.add_argument('--max_new_tokens', type=int, default=2048,
                    help='Maximum new tokens for generation')
args = parser.parse_args()


def predict(args):
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left", cache_dir="local_models/")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model configuration
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        # "attn_implementation": "flash_attention_2" if torch.cuda.is_available() else None,
    }

    if args.lora_path:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, cache_dir="local_models/", **model_kwargs)
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()  # for evaluation, no continue training
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, cache_dir="local_models/", **model_kwargs)

    # Load dataset
    train_dataset, test_dataset = my_load_dataset(args.dataset)
    # test_dataset=test_dataset.select(range(10))
    print(f'Dataset size: {len(test_dataset)}')

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    # Prepare generation config
    generation_config = GenerationConfig(
        temperature=0.2,
        top_p=0.9,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=terminators if "Llama-3" in args.model_path else tokenizer.eos_token_id,
        pad_token_id=128009 if "Llama-3" in args.model_path else tokenizer.pad_token_id,
        do_sample=True,
    )

    # Preprocess all samples
    samples = []
    for i, sample in enumerate(test_dataset):
        if args.dataset in ['gpqa', 'commonsenseQA']:
            ques, options, final_ans = split_sample(sample, args.dataset)
            processed_sample = {
                "id": i,
                "question": ques,
                "options": options,
                "final_answer": final_ans,
                "prompt": build_prompt(ques, options, args.dataset)
                }
        else:
            ques, ration, final_ans = split_sample(sample, args.dataset)
            options = None
            processed_sample = {
                "id": i,
                "question": ques,
                "gold_answer": ration,
                "final_answer": final_ans,
                "prompt": build_prompt(ques, options, args.dataset)
            }
        samples.append(processed_sample)

    res_wrongs = []
    count = 0

    # Batch processing
    with torch.inference_mode():
        for batch_idx in tqdm(range(0, len(samples), args.batch_size), desc="Processing batches"):
            batch = samples[batch_idx:batch_idx + args.batch_size]

            # Process each sample individually first to get original lengths
            individual_inputs = []
            original_lengths = []

            for sample in batch:
                messages = [
                    {"role": "system", "content": think_prompt},
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": sample["prompt"]}
                ]

                # Get individual input without padding
                individual_input = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
                individual_inputs.append(individual_input)
                original_lengths.append(individual_input['input_ids'].shape[1])

            # Now create the batched input with left padding
            messages_batch = [[
                {"role": "system", "content": think_prompt},
                {"role": "user", "content": sample["prompt"]}
            ] for sample in batch]

            # Apply chat template in batch (this will use left padding)
            inputs = tokenizer.apply_chat_template(
                messages_batch,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                padding=True,
            ).to(model.device)

            # Generate responses
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                generation_config=generation_config,
            )

            # Decode responses using the original lengths we stored
            decoded_responses = []
            batch_max_input_length = inputs['input_ids'].shape[1]  # Max length after padding

            for j, (output, original_length) in enumerate(zip(outputs, original_lengths)):
                # With left padding, the generated tokens start after the max input length
                generated_tokens = output[batch_max_input_length:]
                decoded_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                decoded_responses.append(decoded_response)

            # Process batch responses
            for idx, (response, sample) in enumerate(zip(decoded_responses, batch)):
                is_correct = verify_preds(response, sample["final_answer"], args.dataset)
                if is_correct:
                    count += 1
                else:
                    res_wrongs.append(sample | {'prediction': response})
                
    # Save results, extract deepseek_llama3_grpo_gsm8k_combined from checkpoints/deepseek_llama3_grpo_gsm8k_combined/
    model_name = os.path.basename(os.path.normpath(args.lora_path if args.lora_path else args.model_path))
    output_file = f"output/{model_name}_test_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    res_wrongs = sorted(res_wrongs, key=lambda x:int(x['id']))
    with open(output_file, 'w') as f:
        json.dump(res_wrongs, f, indent=4)
    
    # with open("output.json", 'w') as f:
    #     json.dump(count/len(test_dataset), f)

    print(f'For dataset {args.dataset}, {args.lora_path if args.lora_path else args.model_path} Testing Completed!')
    print(f'Accuracy: {count}/{len(test_dataset)} -> {count / len(test_dataset)}')


if __name__ == '__main__':
    predict(args)
