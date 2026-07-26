from tqdm import tqdm
import json
from openai import OpenAI
import argparse
from datasets import load_dataset
from utils import split_sample, cleanup, my_load_dataset, build_prompt
import concurrent.futures
import os


parser = argparse.ArgumentParser(description='Get correct rationales form teachers',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--api', default=os.getenv("DEEPSEEK_API_KEY"),
                    help='DeepSeek API key (defaults to DEEPSEEK_API_KEY)')
parser.add_argument('--dataset', default='gsm8k', choices=['gsm8k', 'math', 'gpqa', 'commonsenseQA'], help="which dataset")
parser.add_argument('--batch_size', '-bs', default=16, type=int, help="batch size")
parser.add_argument('--temperature', default=0.0, type=float)
parser.add_argument('--max_tokens', type=int, default=2048)
args = parser.parse_args()


class TeacherLLMs:
    def __init__(self, args):
        if not args.api:
            raise ValueError("Set DEEPSEEK_API_KEY or pass --api.")
        self.dataset = args.dataset
        self.temperature = args.temperature
        self.max_tokens = args.max_tokens
        self.deepseek = OpenAI(api_key=args.api, base_url="https://api.deepseek.com")
        self.batch_size = args.batch_size
        self.output_file = f'data/{self.dataset}_test_deepseek_answer.jsonl'

    def deepseek_response(self, prompt):
        index, prompt = prompt      # return the index to store in ans
        try:
            response = self.deepseek.chat.completions.create(
                model="deepseek-reasoner",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            rationale, ans = response.choices[0].message.reasoning_content, response.choices[0].message.content
        except BaseException as E:
            print(f"Deepseek Error at index {index}: {E}")
            rationale, ans = None, None
        return index, rationale, ans, prompt

    def collect_correct_rationale(self):
        # check any already processed samples
        processed_indices = set()
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r') as f:
                for line in f:
                    sample = json.loads(line)
                    processed_indices.add(sample["index"])
        
        train_dataset, test_dataset = my_load_dataset(args.dataset)
        train_dataset = test_dataset
        # train_dataset = [train_dataset[i] for i in range(10)]
        
        index_prompts = []
        for idx, sample in enumerate(train_dataset):
            if self.dataset=='gpqa' or self.dataset=='commonsenseQA':
                ques, options, final_ans = split_sample(sample, self.dataset)
                cur_prompt = build_prompt(ques, options, self.dataset)
            else:
                ques, ration, final_ans = split_sample(sample, self.dataset)
                cur_prompt = build_prompt(ques, None, self.dataset)
            # (idx, question)
            index_prompts.append((idx, cur_prompt))
            
        # Filter out already processed samples
        index_prompts = [ip for ip in index_prompts if ip[0] not in processed_indices]
            
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            for index, reasoning_content, content, prompt in tqdm(executor.map(self.deepseek_response, index_prompts), total=len(index_prompts)):
                results.append((index, reasoning_content, content, prompt))

        # Sort results by original dataset order
        results.sort(key=lambda x: x[0])

        with open(self.output_file, "a") as f:
            for index, reasoning_content, content, prompt in results:
                sample = train_dataset[index]
                cur_dict = {
                    'index': index,  # Store the index for future tracking
                    **sample,
                    'prompt': prompt,
                    'deepseek_rationale': reasoning_content,
                    'deepseek_answer': content,
                }

                f.write(json.dumps(cur_dict) + "\n")    # incremental save new result each line
                f.flush()  # Write immediately

        # print(f"For dataset {self.dataset}, infer deepseek answers completed! "
        #       f"The number of correct predictions: {count}/{len(dataset['train'])} -> {count / len(dataset['train'])}")
        print(f"For dataset {self.dataset}, infer deepseek answers completed!")


if __name__ == '__main__':
    teachers = TeacherLLMs(args)
    teachers.collect_correct_rationale()
