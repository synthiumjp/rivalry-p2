"""
extract_answer_tokens_local.py

Drop-in replacement for extract_answer_tokens.py from the H-Neurons pipeline
(Gao et al., 2025). Uses a local instruct model instead of OpenAI GPT-4o.

Same methodology: an LLM identifies answer tokens from the tokenized response.
Same prompt template. Same validation logic. Only difference is the model runs
locally on Apple Silicon via MPS.

Usage:
    python extract_answer_tokens_local.py \
        --input_path data/responses_mistral_1000.jsonl \
        --output_path data/answer_tokens_mistral.jsonl \
        --tokenizer_path mistralai/Mistral-7B-v0.3 \
        --extractor_model google/gemma-3-12b-it
"""

import os
import json
import argparse
from typing import List, Optional, Set

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract answer tokens using a local instruct model."
    )
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to responses JSONL from collect_responses_hf.py"
    )
    parser.add_argument(
        "--output_path", type=str, default="data/answer_tokens.jsonl",
        help="Path to save extracted answer tokens"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True,
        help="Path or name of the TARGET model tokenizer (e.g. mistralai/Mistral-7B-v0.3)"
    )
    parser.add_argument(
        "--extractor_model", type=str, default="google/gemma-3-12b-it",
        help="Local instruct model for answer token extraction"
    )
    parser.add_argument(
        "--max_retries", type=int, default=3,
        help="Number of retries per extraction attempt"
    )
    parser.add_argument(
        "--device", type=str, default="mps",
        help="Device for extractor model (mps, cpu, cuda)"
    )
    return parser.parse_args()


# Prompt template from Gao et al. (2025), unchanged.
USER_INPUT_TEMPLATE = """Question: {question}
Response: {response}
Tokenized Response: {response_tokens}
Please help extract the "answer tokens" from all tokens, removing all redundant information, and the tokens you return must be part of the input Tokenized Response list."""

EXAMPLE_MESSAGES = [
    {
        "role": "user",
        "content": (
            'Question: What is the correct name for the "Flying Lady" ornament '
            "on a Rolls Royce radiator.\n"
            'Response: The correct name for the "Flying Lady" ornament on a '
            "Rolls Royce radiator is the Spirit of Ecstasy.\n"
            "Tokenized Response: ['\u2581The', '\u2581correct', '\u2581name', "
            "'\u2581for', '\u2581the', '\u2581\"', 'F', 'lying', '\u2581Lady', "
            "'\"', '\u2581or', 'nament', '\u2581on', '\u2581a', '\u2581Roll', "
            "'s', '\u2581Roy', 'ce', '\u2581radi', 'ator', '\u2581is', "
            "'\u2581the', '\u2581Spirit', '\u2581of', '\u2581Ec', 'st', "
            "'asy', '.']\n"
            'Please help extract the "answer tokens" from all tokens, '
            "removing all redundant information, and the tokens you return "
            "must form a continuous segment of the input Tokenized Response list."
        ),
    },
    {
        "role": "assistant",
        "content": "['\u2581the', '\u2581Spirit', '\u2581of', '\u2581Ec', 'st', 'asy']",
    },
]


class LocalAnswerTokenExtractor:
    def __init__(self, args):
        self.args = args

        # Target model tokenizer (for tokenizing responses)
        print(f"Loading target tokenizer: {args.tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True
        )

        # Extractor model (local instruct model)
        print(f"Loading extractor model: {args.extractor_model}")
        self.ext_tokenizer = AutoTokenizer.from_pretrained(args.extractor_model)
        self.ext_model = AutoModelForCausalLM.from_pretrained(
            args.extractor_model,
            dtype=torch.float16,
            device_map=args.device,
        )
        self.ext_model.eval()
        print("Extractor model loaded.")

    def get_tokenized_list(self, text: str) -> List[str]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return [self.tokenizer.decode([tid]) for tid in token_ids]

    def extract_via_local_llm(
        self, question: str, response: str, tokens: List[str]
    ) -> Optional[List[str]]:
        """Use local instruct model to select answer tokens from tokenized list."""
        prompt = USER_INPUT_TEMPLATE.format(
            question=question,
            response=response,
            response_tokens=str(tokens),
        )
        messages = EXAMPLE_MESSAGES + [{"role": "user", "content": prompt}]

        for attempt in range(self.args.max_retries):
            try:
                # Two-step: get text, then tokenize. Handles tokenizers
                # that return BatchEncoding from apply_chat_template.
                formatted = self.ext_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.ext_tokenizer(
                    formatted, return_tensors="pt"
                ).to(self.ext_model.device)
                input_len = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    output = self.ext_model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.1,
                        do_sample=True,
                    )

                generated = output[0][input_len:]
                reply = self.ext_tokenizer.decode(
                    generated, skip_special_tokens=True
                ).strip()

                # Parse the reply as a Python list
                reply = reply.replace("\u2018", "'").replace("\u2019", "'")
                reply = reply.replace("'", '"')

                # Extract the list portion if there is surrounding text
                start = reply.find("[")
                end = reply.rfind("]")
                if start != -1 and end != -1:
                    reply = reply[start : end + 1]

                extracted = json.loads(reply)

                # Validation: all selected tokens must exist in original list
                if isinstance(extracted, list) and len(extracted) > 0:
                    if all(t in tokens for t in extracted):
                        return extracted

            except Exception as e:
                if attempt < self.args.max_retries - 1:
                    print(f"  Attempt {attempt + 1} failed: {e}")
                else:
                    print(f"  All {self.args.max_retries} attempts failed for this item.")

        return None

    def load_processed_ids(self) -> Set[str]:
        """Resume from existing output file."""
        if not os.path.exists(self.args.output_path):
            return set()
        ids = set()
        with open(self.args.output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ids.update(json.loads(line).keys())
                except Exception:
                    continue
        return ids

    def run(self):
        processed_ids = self.load_processed_ids()
        total = 0
        extracted = 0
        skipped_judges = 0

        with open(self.args.input_path, "r", encoding="utf-8") as f_in, \
             open(self.args.output_path, "a", encoding="utf-8") as f_out:

            for line in tqdm(f_in, desc="Extracting answer tokens"):
                data = json.loads(line)
                qid = list(data.keys())[0]
                content = data[qid]
                total += 1

                if qid in processed_ids:
                    continue

                # Filter: all judges must agree (same as original)
                judges = content["judges"]
                if len(set(judges)) != 1:
                    skipped_judges += 1
                    continue
                if "uncertain" in judges or "error" in judges:
                    skipped_judges += 1
                    continue

                # Pick the most frequent response as representative
                responses = content["responses"]
                rep_response = max(set(responses), key=responses.count)

                # Tokenize with the TARGET model tokenizer
                tokenized_list = self.get_tokenized_list(rep_response)

                # Extract answer tokens using local LLM
                answer_tokens = self.extract_via_local_llm(
                    content["question"], rep_response, tokenized_list
                )

                if answer_tokens:
                    result = {
                        qid: {
                            "question": content["question"],
                            "response": rep_response,
                            "tokenized_response": tokenized_list,
                            "answer_tokens": answer_tokens,
                            "judge": judges[0],
                        }
                    }
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()
                    extracted += 1

        print(f"\nDone. Total: {total}, Extracted: {extracted}, "
              f"Skipped (judges): {skipped_judges}, "
              f"Failed extraction: {total - skipped_judges - extracted}")


if __name__ == "__main__":
    args = parse_args()
    extractor = LocalAnswerTokenExtractor(args)
    extractor.run()
