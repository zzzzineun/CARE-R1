import os
import argparse
import random
import json
import re

import numpy as np
import torch
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, GenerationConfig
from qwen_vl_utils import process_vision_info


parser = argparse.ArgumentParser(description="Run Qwen2-VL medical evaluation")
parser.add_argument("--modality", required=True)
parser.add_argument("--prompt_type", choices=["simple", "complex", "original"], required=True)
parser.add_argument("--model_path", required=True)
parser.add_argument("--bsz", type=int, default=1)
parser.add_argument("--output_path", required=True)
parser.add_argument("--prompt_path", required=True)
parser.add_argument("--base_path", required=True)
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--do_sample", type=lambda x: str(x).lower() == "true", default=False)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

Modality   = args.modality
Prompt_type = args.prompt_type
MODEL_PATH = args.model_path
BSZ        = args.bsz
OUTPUT_PATH = args.output_path
PROMPT_PATH = args.prompt_path
BASE_PATH  = args.base_path

token = os.environ.get("HF_TOKEN")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map="auto",
    token=token,
)

generation_config = GenerationConfig(
    max_new_tokens=args.max_new_tokens,
    do_sample=args.do_sample,
    temperature=args.temperature,
    num_return_sequences=1,
    pad_token_id=151643,
)

processor = AutoProcessor.from_pretrained(MODEL_PATH, token=token)

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

if Prompt_type == "simple":
    QUESTION_TEMPLATE = """
    {Question}
    Your task: provide the correct single-letter choice (A, B, C, D,...).
    """
elif Prompt_type == "complex":
    QUESTION_TEMPLATE = """
    {Question}
    Your task:
    1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags.
    2. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags.
    3. No extra information or text outside of these tags.
    """
elif Prompt_type == "original":
    QUESTION_TEMPLATE = """
    {Question}
    Think through the question step by step in <think>...</think> tags. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags.
    """

messages = []
for item in data:
    messages.append([{
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{BASE_PATH}/{item['image']}"},
            {"type": "text",  "text": QUESTION_TEMPLATE.format(Question=item["problem"])},
        ],
    }])

all_outputs = []
for i in tqdm(range(0, len(messages), BSZ)):
    batch_messages = messages[i:i + BSZ]

    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
    image_inputs, video_inputs = process_vision_info(batch_messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    generated_ids = model.generate(**inputs, use_cache=True, generation_config=generation_config)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    batch_output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    all_outputs.extend(batch_output_text)


def extract_letter_answer(text):
    match = re.search(r"<answer>\s*([A-Z])\s*</answer>", text)
    return match.group(1) if match else None


final_output = []
correct_number = 0

for input_example, model_output in zip(data, all_outputs):
    ground_truth = extract_letter_answer(input_example["solution"])

    if Prompt_type == "simple":
        model_answer = model_output
    else:
        model_answer = extract_letter_answer(model_output)

    final_output.append({
        "question":         input_example,
        "ground_truth":     ground_truth,
        "model_output":     model_output,
        "extracted_answer": model_answer,
    })

    if model_answer is not None and model_answer == ground_truth:
        correct_number += 1

accuracy = correct_number / len(data) * 100
print(f"\nAccuracy: {accuracy:.2f}%")

with open(OUTPUT_PATH, "w") as f:
    json.dump({"accuracy": accuracy, "results": final_output}, f, indent=2)

print(f"Results saved to {OUTPUT_PATH}")
