# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, Image, concatenate_datasets
from transformers import Qwen2VLForConditionalGeneration

from open_r1.trainer.care_trainer import Qwen2VLCareTrainer
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from PIL import Image as PILImage
from qwen_vl_utils import smart_resize


@dataclass
class CareScriptArguments(ScriptArguments):
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    dataset_selection: str = field(
        default="MR",
        metadata={"help": "Dataset selection. Possible values: 'MR', 'CT', 'XRAY', 'COMBINED'"},
    )
    corr_type: str = field(
        default="black",
        metadata={"help": "Corrupted image type: 'black' or 'blur'."},
    )
    corr_folder: str = field(
        default="images_corr",
        metadata={"help": "Name of the corrupted image folder inside dataset_name/."},
    )


def resize_image_if_needed(image: PILImage.Image, max_pixels: int = 100352, min_pixels: int = 3136) -> PILImage.Image:
    w, h = image.size
    new_h, new_w = smart_resize(h, w, factor=28, min_pixels=min_pixels, max_pixels=max_pixels)
    if (new_h, new_w) == (h, w):
        return image
    return image.resize((new_w, new_h), PILImage.LANCZOS)


def accuracy_reward(completions, solution, image_ids, problems, **kwargs):
    rewards = []

    for comp, sol, image_id, problem in zip(completions, solution, image_ids, problems):
        content = comp[0]["content"]
        reward = 0.0

        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        sol_match = re.search(r"<answer>(.*?)</answer>", sol, flags=re.DOTALL | re.IGNORECASE)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

        content_match = re.search(r"<answer>(.*?)</answer>", content, flags=re.DOTALL | re.IGNORECASE)
        student_answer = content_match.group(1).strip() if content_match else ""

        gt_letter = re.sub(r"\s+", "", ground_truth.upper())
        sa_str    = student_answer.strip()
        sa_upper  = sa_str.upper()

        if not sa_upper:
            reward = 0.0
        else:
            letters_found    = re.findall(r"[A-J]", sa_upper)
            distinct_letters = set(letter.upper() for letter in letters_found)

            if len(letters_found) == 1 and len(distinct_letters) == 1:
                found_letter = distinct_letters.pop()
                if found_letter == gt_letter:
                    leftover = re.sub(r"(?i)" + found_letter, "", sa_str, count=1)
                    reward = 0.5 if leftover.strip() else 1.0
                else:
                    reward = 0.0
            else:
                reward = 0.0

        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Image ID: {image_id}\n")
                f.write(f"Problem: {problem}\n\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format":   format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

QUESTION_TEMPLATE = """
        {Question}
        Your task:
        1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags.
        2. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags.
        3. No extra information or text outside of these tags.
        """


def make_conversation_image(example):
    return {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                ],
            },
        ],
    }


def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    dataset = load_dataset("FreedomIntelligence/Medical_Multimodal_Evaluation_Data", split="test")

    dataset_mr    = dataset.filter(lambda x: x["subset"] == "MR (Mag-netic Resonance Imaging)").select(range(600))
    dataset_ct    = dataset.filter(lambda x: x["subset"] == "CT(Computed Tomography)").select(range(600))
    dataset_xray  = dataset.filter(lambda x: x["subset"] == "X-Ray").select(range(600))
    dataset_ultra = dataset.filter(lambda x: x["subset"] == "ultrasound").select(range(600))
    dataset_dermos= dataset.filter(lambda x: x["subset"] == "Dermoscopy").select(range(600))
    dataset_fund  = dataset.filter(lambda x: x["subset"] == "Fundus Photography").select(range(600))
    dataset_oct   = dataset.filter(lambda x: x["subset"] == "OCT (Optical Coherence Tomography").select(range(570))
    dataset_micro = dataset.filter(lambda x: x["subset"] == "Microscopy Images").select(range(600))

    if script_args.dataset_selection == "MR":
        dataset = dataset_mr
    elif script_args.dataset_selection == "CT":
        dataset = dataset_ct
    elif script_args.dataset_selection == "XRAY":
        dataset = dataset_xray
    elif script_args.dataset_selection == "Ultrasound":
        dataset = dataset_ultra
    elif script_args.dataset_selection == "Dermoscopy":
        dataset = dataset_dermos
    elif script_args.dataset_selection == "Fundus":
        dataset = dataset_fund
    elif script_args.dataset_selection == "OCT":
        dataset = dataset_oct
    elif script_args.dataset_selection == "Microscopy":
        dataset = dataset_micro
    elif script_args.dataset_selection == "COMBINED":
        dataset = concatenate_datasets([dataset_mr, dataset_ct, dataset_xray])
    else:
        raise ValueError(f"Invalid dataset_selection: {script_args.dataset_selection}.")

    dataset = dataset.shuffle(seed=42)
    print("dataset Huatuo loaded")

    def create_problem_solution(example, base_image_path):
        letter_map = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        labeled_options = [f"{letter_map[i]}) {opt}" for i, opt in enumerate(example["options"])]
        example["problem"]  = example["question"] + "\n" + "\n".join(labeled_options)
        correct_index       = example["options"].index(example["answer"])
        example["solution"] = letter_map[correct_index]
        example["image_id"] = example["image"][0]
        image_path          = os.path.join(base_image_path, example["image"][0])
        img = PILImage.open(image_path).convert("RGB")
        example["image"] = resize_image_if_needed(img)
        return example

    base_image_path = script_args.dataset_name
    dataset = dataset.map(lambda x: create_problem_solution(x, base_image_path), num_proc=4)
    dataset = dataset.remove_columns(["question", "dataset", "subset", "answer"])
    dataset = dataset.cast_column("image", Image())

    if "image" in dataset.features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)

    trainer = Qwen2VLCareTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        corr_type=script_args.corr_type,
        corr_folder=script_args.corr_folder,
        base_image_path=script_args.dataset_name,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((CareScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)