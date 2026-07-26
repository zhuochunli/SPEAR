import json
import torch
import argparse
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from peft import LoraConfig, get_peft_model, PeftModel
from data import rl_preprocess_dataset, rl_make_conversation
from reward import combined_reward, format_reward, accuracy_reward
from trl import GRPOTrainer, GRPOConfig
import wandb
import os

# Multi-GPU communication settings
os.environ["VLLM_HOST_IP"] = "127.0.0.1"

train_parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
train_parser.add_argument('--model_path', '-m', default='meta-llama/Meta-Llama-3-8B-Instruct')
train_parser.add_argument('--lora_path', default=None)
train_parser.add_argument('--answer_path', '-ap', default='data/gsm8k_deepseek_answer_prompt.jsonl')
train_parser.add_argument('--hf_token', '-ht', default=os.getenv("HF_TOKEN"),
                          help='Hugging Face access token (defaults to HF_TOKEN)')
train_parser.add_argument('--output_path', '-op', default="checkpoints/llama3_grpo_gsm8k")
train_parser.add_argument('--task_type', default='gsm8k', choices=["gsm8k", "math", "gpqa", "commonsenseQA"])
train_parser.add_argument('--algorithm', default='grpo', choices=["grpo", "dr_grpo", "dapo"],
                          help="grpo: standard GRPO\ndr_grpo: Dr. GRPO (length + difficulty bias fix)\ndapo: DAPO loss")
train_parser.add_argument('--learning_rate', '-lr', default=5e-6, type=float)
# train_parser.add_argument('--max_prompt_length', default=256, type=int)
train_parser.add_argument('--max_comp_length', default=1024, type=int)
train_parser.add_argument('--epochs', '-e', default=1, type=int)
train_parser.add_argument('--batch_size', '-bs', default=8, type=int)
train_parser.add_argument('--beta', default=0.04, type=float)
train_parser.add_argument('--seed', type=int, default=42)
train_parser.add_argument('--baseline', action='store_true', help='Use format + accuracy rewards only')
args = train_parser.parse_args()

os.environ["WANDB_PROJECT"] = "deep_distill"

# Algorithm-specific config overrides
ALGO_CONFIGS = {
    "grpo": {
        "loss_type": "grpo",
        "scale_rewards": "group",   # default
        "num_generations": 4,
        "beta": args.beta,
    },
    "dr_grpo": {
        "loss_type": "dr_grpo",
        "scale_rewards": "none",    # disables question-level difficulty bias
        "num_generations": 4,
        "beta": args.beta,
    },
    "dapo": {
        "loss_type": "dapo",
        "scale_rewards": "group",
        "num_generations": 4,
        "beta": args.beta,        # DAPO typically uses 0.1
    },
}


class RewardCallback(TrainerCallback):
    def __init__(self, baseline_mode=False):
        self.last_seen_reward = 0.0
        self.watch_key = "accuracy_reward" if baseline_mode else "combined_reward"

    def on_log(self, args, state, control, logs=None, **kwargs):
        reward_keys = [f"eval_rewards/{self.watch_key}", f"rewards/{self.watch_key}", "eval_reward"]
        for key in reward_keys:
            if logs and key in logs:
                self.last_seen_reward = logs[key]

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        reward = metrics.get(f"eval_rewards/{self.watch_key}")
        if reward is None:
            reward = self.last_seen_reward
        metrics["eval_best_model_metric"] = reward
        print(f"\n[Callback] Metric Sync -> best_model_metric: {reward:.4f}")


class CustomTrainer:
    def __init__(self, args):
        self.args = args
        algo_cfg = ALGO_CONFIGS[args.algorithm]

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            token=args.hf_token,
            cache_dir='local_models/',
        )
        model.gradient_checkpointing_enable()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side='left')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = rl_preprocess_dataset(args.answer_path, args.task_type, args.seed)
        dataset = dataset.map(rl_make_conversation)
        dataset = dataset.map(lambda x: {"task_type": args.task_type})
        # GPQA: 0.03 for eval, ~1000 training samples
        split_dataset = dataset.train_test_split(test_size=0.03 if args.task_type == 'gpqa' else 0.01, seed=args.seed)

        if args.lora_path:
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

        training_args = GRPOConfig(
            output_dir=args.output_path,
            learning_rate=args.learning_rate,
            remove_unused_columns=False,
            per_device_train_batch_size=args.batch_size,
            gradient_checkpointing=True,
            gradient_accumulation_steps=4,
            num_train_epochs=args.epochs,
            bf16=True,
            beta=algo_cfg["beta"],
            loss_type=algo_cfg["loss_type"],
            scale_rewards=algo_cfg["scale_rewards"],
            num_generations=algo_cfg["num_generations"],
            max_completion_length=args.max_comp_length,
            seed=args.seed,
            run_name=args.output_path,
            report_to=["wandb"],
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,     # 50 for gpqa
            eval_strategy="steps",
            eval_steps=100,
            per_device_eval_batch_size=args.batch_size,
            load_best_model_at_end=True,
            metric_for_best_model="best_model_metric",
            greater_is_better=True,
            save_total_limit=2,
        )

        selected_rewards = [format_reward, accuracy_reward] if args.baseline else [combined_reward]
        self.reward_callback = RewardCallback(baseline_mode=args.baseline)

        self.trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=selected_rewards,
            args=training_args,
            train_dataset=split_dataset["train"],
            eval_dataset=split_dataset["test"],
            callbacks=[self.reward_callback],
        )

    def train(self):
        print(f"Starting {self.args.algorithm.upper()} training...")
        self.trainer.train()
        self.trainer.save_model(self.args.output_path)
        print(f"{self.args.algorithm.upper()} training complete. Saved at: {self.args.output_path}")


if __name__ == '__main__':
    trainer = CustomTrainer(args)
    trainer.train()
