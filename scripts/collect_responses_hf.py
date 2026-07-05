import os
import json
import re
import string
import argparse
import time
from typing import List, Set, Dict

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI


def parse_args():
    parser = argparse.ArgumentParser(description="Consistency Filtering with Rule or LLM Judge.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model for sampling")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the TriviaQA parquet file")
    parser.add_argument("--output_path", type=str, default="data/consistency_samples.jsonl", help="Output path")
    parser.add_argument("--sample_num", type=int, default=10, help="Samples per question")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of questions to process")
    parser.add_argument("--judge_type", type=str, choices=["rule", "llm"], default="rule", help="How to judge correctness")
    parser.add_argument("--api_key", type=str, default=None, help="API key for LLM Judge")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1", help="API base URL")
    parser.add_argument("--judge_model", type=str, default="gpt-4o", help="Model name for LLM Judge")
    return parser.parse_args()


def normalize_answer(s: str) -> str:
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def handle_punc(text):
        exclude = set(string.punctuation + "''`")
        return ''.join(ch if ch not in exclude else ' ' for ch in text)
    if not s: return ""
    return white_space_fix(remove_articles(handle_punc(str(s).lower().replace('_', ' ')))).strip()


def load_existing_qids(path: str) -> Set[str]:
    if not os.path.exists(path): return set()
    qids = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                qids.update(data.keys())
            except: continue
    return qids


class ConsistencySampler:
    def __init__(self, args):
        self.args = args
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple Silicon MPS backend")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA backend")
        else:
            self.device = torch.device("cpu")
            print("Using CPU backend")

        print(f"Loading model: {args.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.float16, trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        print("Model loaded.")

        self.judge_client = None
        if args.judge_type == "llm":
            if not args.api_key:
                raise ValueError("API Key is required for LLM Judge.")
            self.judge_client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    def generate_response(self, messages):
        # Try chat template first, fall back to raw text for base models
        try:
            input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            input_text = messages[0]["content"]
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=50, temperature=1.0, top_p=0.9, top_k=50,
                do_sample=True, pad_token_id=self.tokenizer.pad_token_id
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def rule_judge(self, response, norm_gts):
        norm_res = normalize_answer(response)
        for gt in norm_gts:
            if gt and gt in norm_res: return "true"
        return "false"

    def llm_judge(self, question, response, answer_list):
        prompt = (
            f"Question: {question}\nResponse: {response}\nCorrect Answers: {answer_list}\n"
            f"Please judge whether the response is correct or not. "
            f"Return 't' if correct, 'f' if incorrect. No additional information."
        )
        for attempt in range(5):
            try:
                completion = self.judge_client.chat.completions.create(
                    model=self.args.judge_model,
                    messages=[{"role": "user", "content": prompt}], temperature=0.0
                )
                res = completion.choices[0].message.content.strip().lower()
                if 't' in res: return "true"
                if 'f' in res: return "false"
            except Exception as e:
                print(f"Judge API failed (attempt {attempt+1}): {e}")
                time.sleep(1)
        return "error"

    def process_data(self):
        dataset = load_dataset("parquet", data_files=self.args.data_path, split="train")
        if self.args.max_samples:
            dataset = dataset.select(range(self.args.max_samples))
        processed_qids = load_existing_qids(self.args.output_path)
        all_correct_count = 0
        all_incorrect_count = 0

        with open(self.args.output_path, 'a', encoding='utf-8') as f:
            for item in tqdm(dataset, desc=f"Sampling ({self.args.judge_type} judge)"):
                qid = str(item.get('question_id', ''))
                if qid in processed_qids: continue
                question = item.get('question', '')
                if not question or 'answer' not in item: continue

                raw_aliases = []
                for col in ['aliases', 'normalized_aliases']:
                    val = item['answer'].get(col)
                    if val: raw_aliases.extend(val) if isinstance(val, list) else raw_aliases.append(str(val))
                norm_gts = [normalize_answer(a) for a in set(raw_aliases) if a]
                if not norm_gts: continue

                suffix = "Respond with the answer only, without any explanation."
                messages = [{"role": "user", "content": f"{question.strip()} {suffix}"}]
                responses = []
                judges = []
                judge_cache = {}

                for _ in range(self.args.sample_num):
                    try:
                        ans = self.generate_response(messages)
                        responses.append(ans)
                        uncertain_terms = ["don't know", "cannot", "not provided", "no information"]
                        if any(term in ans.lower() for term in uncertain_terms):
                            judges.append("uncertain")
                            continue
                        if self.args.judge_type == "rule":
                            judges.append(self.rule_judge(ans, norm_gts))
                        else:
                            if ans not in judge_cache:
                                judge_cache[ans] = self.llm_judge(question, ans, raw_aliases)
                            judges.append(judge_cache[ans])
                    except Exception as e:
                        print(f"Sampling error at {qid}: {e}")
                        break

                if len(responses) < self.args.sample_num: continue
                true_count = judges.count("true")
                if true_count == self.args.sample_num: all_correct_count += 1
                elif true_count == 0: all_incorrect_count += 1

                result = {qid: {
                    "question": f"{question.strip()} {suffix}",
                    "responses": responses, "judges": judges,
                    "ground_truth": list(set(raw_aliases))
                }}
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                if len(processed_qids) % 10 == 0:
                    tqdm.write(f"Stats -> All-Correct: {all_correct_count}, All-Incorrect: {all_incorrect_count}")


if __name__ == "__main__":
    args = parse_args()
    sampler = ConsistencySampler(args)
    sampler.process_data()
