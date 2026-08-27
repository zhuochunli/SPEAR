import json
import torch
import argparse
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TrainerCallback
from peft import LoraConfig, get_peft_model, PeftModel
from data import rl_preprocess_dataset, rl_make_conversation
from reward import combined_reward, format_reward, accuracy_reward
from torch import nn
from trl import GRPOTrainer, GRPOConfig
from all_prompts import think_prompt
import wandb
import os


train_parser = argparse.ArgumentParser(description='train student model', formatter_class=argparse.RawTextHelpFormatter)
train_parser.add_argument('--model_path', '-m', default='meta-llama/Meta-Llama-3-8B-Instruct', help='the student model path')
train_parser.add_argument('--lora_path', default=None, help='Previous trained LoRA adapter path')
train_parser.add_argument('--answer_path', '-ap', default='data/gsm8k_deepseek_answer_prompt.jsonl', help='the teacher model answer path')
train_parser.add_argument('--hf_token', '-ht', default=os.getenv("HF_TOKEN"),
                          help='Hugging Face access token (defaults to HF_TOKEN)')
train_parser.add_argument('--output_path', '-op', default="checkpoints/gold_llama3_dr_grpo_gsm8k", help='model output path')
train_parser.add_argument('--task_type', default='gsm8k', choices=["gsm8k", "math", "gpqa", "commonsenseQA"], help='the type of reasoning task')
train_parser.add_argument('--rewards', nargs='+', choices=["format", "accuracy", "self_consistency_reward", "document", "sentence", "cue_word"], default=["format", "accuracy"], help="select rewards function")
train_parser.add_argument('--learning_rate', '-lr', default=5e-6, type=float, help='learning rate')
train_parser.add_argument('--max_prompt_length', default=128, type=int, help='max prompt length')
train_parser.add_argument('--max_comp_length', default=2048, type=int, help='max completion length')
train_parser.add_argument('--epochs', '-e', default=1, type=int, help="num_train_epochs")
train_parser.add_argument('--batch_size', '-bs', default=8, type=int, help="batch size")
train_parser.add_argument('--beta', default=0.04, type=float, help="beta value")
train_parser.add_argument('--seed', type=int, default=731, help='seed setting')
train_parser.add_argument('--baseline', action='store_true', help='If true, use format + accuracy rewards only')
args = train_parser.parse_args()

os.environ["WANDB_PROJECT"] = "deep_distill"

class RewardCallback(TrainerCallback):
    def __init__(self, baseline_mode=False):
        self.last_seen_reward = 0.0
        # Determine which key to watch based on the mode
        self.watch_key = "accuracy_reward" if baseline_mode else "combined_reward"

    def on_log(self, args, state, control, logs=None, **kwargs):
        # TRL logs list-based rewards as rewards/function_name
        reward_keys = [f"eval_rewards/{self.watch_key}", f"rewards/{self.watch_key}", "eval_reward"]
        for key in reward_keys:
            if logs and key in logs:
                self.last_seen_reward = logs[key]

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        reward = metrics.get(f"eval_rewards/{self.watch_key}")
        if reward is None:
            reward = self.last_seen_reward

        # Use a consistent key for 'metric_for_best_model'
        metrics["eval_best_model_metric"] = reward
        print(f"\n[Callback] Metric Sync -> best_model_metric: {reward:.4f}")

class CustomTrainer:
    def __init__(self, args):
        self.args = args
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            # device_map="auto",
            token=args.hf_token,
            cache_dir='local_models/',
        )
        # model.gradient_checkpointing_enable()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side='left')

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  
        
        # whether generate sentence embeddings for teacher thinking process
        dataset = rl_preprocess_dataset(args.answer_path, args.task_type, args.seed)
        dataset = dataset.map(rl_make_conversation)
        dataset = dataset.map(lambda x: {"task_type": args.task_type})
        
        # Reserve 5% of the training data for development/model selection.
        split_dataset = dataset.train_test_split(test_size=0.05, seed=args.seed)
        train_ds = split_dataset["train"]
        eval_ds = split_dataset["test"]
        
        if args.lora_path:   # continue training
            model = PeftModel.from_pretrained(model, args.lora_path, is_trainable=True)
        else:
            lora_config = LoraConfig(
                task_type="CAUSAL_LM",
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],
            )
            model = get_peft_model(model, lora_config)
            
        model.train()
        
        # Configure training arguments using GRPOConfig
        training_args = GRPOConfig(
            output_dir=args.output_path,
            learning_rate=args.learning_rate,
            remove_unused_columns=False,  # to access the solution column in accuracy_reward
            per_device_train_batch_size=args.batch_size,
            gradient_checkpointing=True,
            gradient_accumulation_steps=4,
            num_train_epochs=args.epochs,
            bf16=True,
            beta=args.beta,    # default: 0.04
            
            # --- Dr. GRPO Adjustments ---
            loss_type="dr_grpo",   # Mitigates response length bias
            scale_rewards="none",   # Mitigates question-level difficulty bias
            # ----------------------------

            num_generations=8,  # default: 8
            # max_prompt_length=args.max_prompt_length,  # no longer exists in TRL 0.29.0
            max_completion_length=args.max_comp_length,  # default: 2048
            seed=args.seed,
            run_name=args.output_path,
            report_to=["wandb"],
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,     # 50 for gpqa
            eval_strategy="steps",
            eval_steps=100,    # 50 for gpqa
            per_device_eval_batch_size=args.batch_size,
            load_best_model_at_end=True, 
            metric_for_best_model="best_model_metric",
            greater_is_better=True,      
            save_total_limit=2,          
        )
        
        if args.baseline:
            selected_rewards = [format_reward, accuracy_reward]
        else:
            selected_rewards = [combined_reward]
    
        self.reward_callback = RewardCallback(baseline_mode=args.baseline)
        self.trainer = GRPOTrainer(model=model, 
                                   processing_class=tokenizer,
                                   reward_funcs=selected_rewards,
                                   args=training_args, 
                                   train_dataset=train_ds,
                                   eval_dataset=eval_ds,
                                   callbacks=[self.reward_callback],
                                  )
    
    def train(self):
        print("Starting Dr. GRPO training...")
        self.trainer.train()
        self.trainer.save_model(self.args.output_path)
        print("Dr. GRPO Training complete. Model saved at: ", self.args.output_path)

if __name__ == '__main__':
    trainer = CustomTrainer(args)
    trainer.train()
