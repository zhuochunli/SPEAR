import json
import torch
import argparse
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, default_data_collator
from peft import LoraConfig, get_peft_model
from data import preprocess_dataset, tokenize_function
from torch import nn
from accelerate import Accelerator
from trl import SFTTrainer, SFTConfig
from all_prompts import think_prompt
import math
import wandb
import os

# meta-llama/Meta-Llama-3-8B-Instruct
# Qwen/Qwen2.5-1.5B-Instruct
train_parser = argparse.ArgumentParser(description='train student model', formatter_class=argparse.RawTextHelpFormatter)
train_parser.add_argument('--model_path', default='meta-llama/Meta-Llama-3-8B-Instruct', help='the student model path')
train_parser.add_argument('--answer_path', default='data/gsm8k_deepseek_answer_prompt.jsonl', help='the teacher model answer path')
train_parser.add_argument('--flash_attention', action='store_true', help='whether to use flash_attention')
train_parser.add_argument('--hf_token', '-ht', default=os.getenv("HF_TOKEN"),
                          help='Hugging Face access token (defaults to HF_TOKEN)')
train_parser.add_argument('--output_path', default="checkpoints/deepseek_llama3_sft_gsm8k", help='model output path')
train_parser.add_argument('--learning_rate', '-lr', default=2e-5, type=float, help='learning rate')
train_parser.add_argument('--max_seq_length', default=2048, type=int, help='max_seq_length')
train_parser.add_argument('--epochs', default=1, type=int, help="num_train_epochs")
train_parser.add_argument('--batch_size', '-bs', default=4, type=int, help="batch size")
train_parser.add_argument('--seed', default=731, type=int, help='seed setting')
args = train_parser.parse_args()


os.environ["WANDB_PROJECT"] = "deep_distill"
wandb.init(name=args.output_path)

    
class PerTokenSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")

        # Safety check: ensure input_ids and labels are the same length
        # If SFTTrainer truncated input_ids, we must truncate labels here too
        if labels.shape[1] != inputs["input_ids"].shape[1]:
            labels = labels[:, :inputs["input_ids"].shape[1]]

        outputs = model(**inputs)
        logits = outputs.logits

        # Shift logic
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss
    

class CustomTrainer:
    def __init__(self, args):
        self.args = args
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            # use_flash_attention_2=args.flash_attention,
            attn_implementation="flash_attention_2" if args.flash_attention else "sdpa",
            # device_map="auto",
            cache_dir='local_models/',
            token=args.hf_token,
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.padding_side = "right"

        # if 'Llama-3' in args.model_path:  # add pad_token
        #     # model.config.pad_token_id = 128002  # "<|reserved_special_token_0|>"
        #     tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        dataset = preprocess_dataset(args.answer_path, args.seed)
        self.tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer, max_length=args.max_seq_length), remove_columns=dataset["train"].column_names, batched=False)
        # Convert to torch tensors
        self.tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        sample = self.tokenized_dataset["train"][0]

        # LoRA config based on QLoRA paper
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=8,
            # bias="all",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],  # Target attention layers
        )
        # prepare model for training
        model = get_peft_model(model, peft_config)
        # model.to("cuda")
        
        train_args = SFTConfig(
            output_dir=args.output_path,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size if args.flash_attention else args.batch_size // 2,
            per_device_eval_batch_size=args.batch_size if args.flash_attention else args.batch_size // 2,
            gradient_accumulation_steps=4,
            # gradient_checkpointing=True,
            logging_steps=50,
            save_strategy="epoch",
            learning_rate=args.learning_rate,
            bf16=True,
            fp16=False,
            max_grad_norm=1.0,
            warmup_ratio=0.03,
            weight_decay=0.01,  # L2-regularization
            lr_scheduler_type="linear",
            seed=args.seed,
            save_total_limit=2,  # best one and last ones
            eval_strategy="epoch",
            # evaluation_strategy="steps",
            # eval_steps=2,
            load_best_model_at_end=True,
            report_to=["wandb"],
            max_length=args.max_seq_length, 
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True}
        )

        # data_collator = DataCollatorForLanguageModeling(tokenizer,mlm=False,pad_to_multiple_of=None)

        # self.trainer = Trainer(model=model,
        #                        args=train_args,
        #                        train_dataset=tokenized_dataset["train"],
        #                        eval_dataset=tokenized_dataset["validation"],
        #                        data_collator=default_data_collator,
        #                        tokenizer=tokenizer,
        #                        )
        
        # formatting_func = build_formatting_func(tokenizer)
        # ensure tokenized inputs are used directly
        # train_args.remove_unused_columns = False
        self.trainer = PerTokenSFTTrainer(model=model,
                               args=train_args,
                               train_dataset=self.tokenized_dataset["train"],
                               eval_dataset=self.tokenized_dataset["validation"],
                               # packing=True,
                               # peft_config=peft_config,
                               processing_class=tokenizer,
                               # formatting_func=formatting_func,
                               data_collator=default_data_collator,
                                 )
    
    def train(self):
        # train
        print("Starting training...")
        self.trainer.train()
        # save model
        self.trainer.save_model()
        print("Training complete. Model saved at: ", self.args.output_path)


if __name__ == '__main__':
    trainer = CustomTrainer(args)
    trainer.train()
