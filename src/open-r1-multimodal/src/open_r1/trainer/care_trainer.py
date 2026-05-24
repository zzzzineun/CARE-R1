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
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import math
import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from PIL import Image as PILImage
from qwen_vl_utils import smart_resize
from transformers import (
    AriaForConditionalGeneration,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
from datetime import datetime

import re


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class Qwen2VLCareTrainer(Trainer):
    """
    Trainer combining tgtmask (corrupted-image gap weighting) with training improvements.

    tgtmask idea:
    - Generate completions from both original and corrupted images
    - Compute reward gap = mean_acc(original) - mean_acc(corrupted)
    - Scale advantages by w = 1 + exp_scaled_gap to amplify visually informative samples

    Additional improvements over standard GRPO:
    1. Dynamic Sampling: exclude groups where std(reward) ≈ 0
    2. Token-level Loss: normalize by total valid tokens, not per-seq
    3. Overlong Filtering: zero-out sequences truncated at max length
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        # ── tgtmask 파라미터 ────────────────────────────────────────────────
        corr_type: str = "black",
        corr_folder: str = "images_corr",
        base_image_path: str = "",
    ):
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-tgtmask")

        self.corr_type = corr_type
        self.corr_folder = corr_folder
        self.base_image_path = base_image_path

        # ── Model Load ─────────────────────────────────
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass
            elif isinstance(torch_dtype, str):
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id or "Huatuo" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # ── Processing Class ─────────────────────────────────
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id or "Huatuo" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # ── Reward Functions ─────────────────────────────────
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # ── Reward processing class ─────────────────────────────────
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # ── Data collator ─────────────────────────────────
        def data_collator(features):
            return features

        # ── Training arguments ─────────────────────────────────
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1,
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )

        model.warnings_issued["estimate_tokens"] = True
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.model_accepts_loss_kwargs = False

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        # ── Logging Config ─────────────────────────────────
        if os.getenv("DEBUG_MODE") == "true" and self.accelerator.is_main_process:
            log_path = os.getenv("LOG_PATH")
            if log_path:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write("=" * 60 + "\n")
                    f.write(f"  Training Configuration (tgtmask)\n")
                    f.write("=" * 60 + "\n")
                    f.write(f"  Start Time         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"  Dataset            : {os.getenv('DATASET', 'N/A')}\n")
                    f.write(f"  Dataset Selection  : {os.getenv('DATASET_SELECTION', 'N/A')}\n")
                    f.write(f"  Model              : {model_id}\n")
                    f.write(f"  Output Dir         : {args.output_dir}\n")
                    f.write(f"  Num Generations    : {self.num_generations}\n")
                    f.write(f"  Num Epochs         : {args.num_train_epochs}\n")
                    f.write(f"  Batch Size/Device  : {args.per_device_train_batch_size}\n")
                    f.write(f"  Grad Accum Steps   : {args.gradient_accumulation_steps}\n")
                    f.write(f"  Max Prompt Len     : {self.max_prompt_length}\n")
                    f.write(f"  Max Completion Len : {self.max_completion_length}\n")
                    f.write(f"  Learning Rate      : {args.learning_rate}\n")
                    f.write(f"  Reward Functions   : {[rf.__name__ if callable(rf) else rf for rf in reward_funcs]}\n")
                    f.write(f"  Corr Type          : {corr_type}\n")
                    f.write(f"  Corr Folder        : {corr_folder}\n")
                    f.write(f"  Base Image Path    : {base_image_path}\n")
                    f.write(f"  LOG_PATH           : {log_path}\n")
                    f.write("=" * 60 + "\n\n")


    # ── Load pre-generated corrupted images from corr_folder ─────────────────
    @staticmethod
    def resize_image_if_needed(image: PILImage.Image, max_pixels: int = 100352, min_pixels: int = 3136) -> PILImage.Image:
        w, h = image.size
        new_h, new_w = smart_resize(h, w, factor=28, min_pixels=min_pixels, max_pixels=max_pixels)
        if (new_h, new_w) == (h, w):
            return image
        return image.resize((new_w, new_h), PILImage.LANCZOS)

    def load_corrupted_images(self, images: list, image_ids: list) -> list:
        result = []
        for img, img_id in zip(images, image_ids):
            fname = os.path.basename(img_id)
            stem, ext = os.path.splitext(fname)
            corr_fname = f"{stem}_{self.corr_type}{ext}"
            corr_path = os.path.join(
                self.base_image_path, self.corr_folder, corr_fname
            )

            if os.path.exists(corr_path):
                corr_img = PILImage.open(corr_path).convert("RGB")
                corr_img = corr_img.resize(img.size, PILImage.LANCZOS)
            else:
                corr_img = PILImage.new("RGB", img.size, (0, 0, 0))
                if os.getenv("DEBUG_MODE") == "true" and self.accelerator.is_main_process:
                    log_path = os.getenv("LOG_PATH")
                    if log_path:
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(f"[WARN] Corrupted image not found, using black fallback: {corr_path}\n")

            result.append(corr_img)
        return result

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw):
        logits = model(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        ).logits                                    # (B, L, V)
        logits    = logits[:, :-1, :]               # (B, L-1, V)
        input_ids = input_ids[:, 1:]                # (B, L-1)
        log_probs = logits.log_softmax(dim=-1)      # (B, L-1, V)
        return torch.gather(
            log_probs, dim=2, index=input_ids.unsqueeze(2)
        ).squeeze(2)                                # (B, L-1)

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    # ──────────────────────────────────────────────────────────────────
    # Main training step
    # ──────────────────────────────────────────────────────────────────
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("Returning outputs is not supported")

        # ── [1] Extract inputs ─────────────────────────────────────────
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        images = [x["image"] for x in inputs]
        image_ids = [x["image_id"] for x in inputs]
        problems = [x["problem"] for x in inputs]
        batch_size = len(inputs)

        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # ── [2] Original image generation ─────────────────────────────
        num_generations = self.generation_config.num_return_sequences

        image_ids = [x for x in image_ids for _ in range(num_generations)]   # B*G
        problems  = [x for x in problems  for _ in range(num_generations)]   # B*G

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(
                **prompt_inputs,
                generation_config=self.generation_config,
            )   # (B*G, L)

        prompt_length  = prompt_ids.size(1)
        prompt_ids     = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]
        prompt_mask    = prompt_mask.repeat_interleave(num_generations, dim=0)

        # ── [3] Corrupted image generation ────────────────────────────
        batch_image_ids = image_ids[::num_generations]
        masked_images = self.load_corrupted_images(images, batch_image_ids)

        masked_img_out = self.processing_class.image_processor(
            images=masked_images,
            return_tensors="pt",
        )
        masked_img_out = super()._prepare_inputs(masked_img_out)

        masked_gen_inputs = {
            "input_ids":      prompt_inputs["input_ids"],
            "attention_mask": prompt_inputs["attention_mask"],
            "pixel_values":   masked_img_out["pixel_values"],
            "image_grid_thw": masked_img_out["image_grid_thw"],
        }
        masked_full_prompt_length = prompt_inputs["input_ids"].size(1)

        with torch.no_grad():
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                masked_prompt_completion_ids = unwrapped_model.generate(
                    **masked_gen_inputs,
                    generation_config=self.generation_config,
                )   # (B*G, L)
                masked_completion_ids_tensor = masked_prompt_completion_ids[:, masked_full_prompt_length:]

        # ── [4] EOS masking + attention mask for original ──────────────
        is_eos = completion_ids == self.processing_class.eos_token_id
        device  = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask  = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # ── Overlong Filtering ─────────────────────────────────────────
        is_overlong = ~is_eos.any(dim=1)
        completion_mask = completion_mask * (~is_overlong).unsqueeze(1).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        pixel_values_rep   = prompt_inputs["pixel_values"][None].repeat_interleave(num_generations, dim=0)
        image_grid_thw_rep = prompt_inputs["image_grid_thw"].repeat_interleave(num_generations, dim=0)

        # ── [5] Per-token log-probs (with grad) ───────────────────────
        per_token_logps = self._get_per_token_logps(
            model, prompt_completion_ids, attention_mask, pixel_values_rep, image_grid_thw_rep
        )
        per_token_logps = per_token_logps[:, prompt_length - 1:]

        # ── [6] Decode original completions ───────────────────────────
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": c}] for c in completions]

        # ── [7] Expand prompts, build reward_kwargs ────────────────────
        prompts = [p for p in prompts for _ in range(num_generations)]   # B*G

        reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
        for key in reward_kwargs:
            for example in inputs:
                reward_kwargs[key].extend([example[key]] * num_generations)

        # ── [8] Original reward computation ───────────────────────────
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                out = reward_func(
                    prompts=prompts, completions=completions,
                    image_ids=image_ids, problems=problems,
                    **reward_kwargs,
                )
                rewards_per_func[:, i] = torch.tensor(out, dtype=torch.float32, device=device)

        rewards = rewards_per_func.sum(dim=1)   # (B*G,)

        # ── [9] Corrupted reward computation + gap/weight ─────────────
        masked_completions_text = self.processing_class.batch_decode(
            masked_completion_ids_tensor, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            masked_completions = [
                [{"role": "assistant", "content": c}] for c in masked_completions_text
            ]
        else:
            masked_completions = masked_completions_text

        masked_rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, masked_completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, masked_completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    masked_rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                out = reward_func(
                    prompts=prompts, completions=masked_completions,
                    image_ids=image_ids, problems=problems,
                    **reward_kwargs,
                )
                masked_rewards_per_func[:, i] = torch.tensor(out, dtype=torch.float32, device=device)

        masked_rewards = masked_rewards_per_func.sum(dim=1)   # (B*G,)

        # Per-sample gap and weight — accuracy reward ONLY (rewards_per_func[:, 0])
        acc_rewards_orig = rewards_per_func[:, 0]
        acc_rewards_mask = masked_rewards_per_func[:, 0]
        mean_reward_orig = acc_rewards_orig.view(batch_size, num_generations).mean(dim=1)
        mean_reward_mask = acc_rewards_mask.view(batch_size, num_generations).mean(dim=1)
        gap = mean_reward_orig - mean_reward_mask

        beta = 1.0
        scaled_gap = (torch.exp(beta * gap) - 1) / (math.exp(beta * 1) - 1)
        scaled_gap = torch.clamp(scaled_gap, min=0.0, max=1.0)
        w = 1.0 + scaled_gap   # range: [1, 2]

        w_expanded = w.repeat_interleave(num_generations, dim=0)

        # ── [10] Advantage computation ─────────────────────────────────
        mean_grouped_rewards = rewards.view(batch_size, num_generations).mean(dim=1)
        std_grouped_rewards  = rewards.view(batch_size, num_generations).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
        std_grouped_rewards  = std_grouped_rewards.repeat_interleave(num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        advantages = w_expanded * advantages

        # ── Dynamic Sampling: reward 분산=0인 그룹 제외 ──────────────────────────────────
        std_grouped_for_filter = rewards.view(batch_size, num_generations).std(dim=1)
        valid_groups = (std_grouped_for_filter > 1e-4)
        valid_mask = valid_groups.repeat_interleave(num_generations, dim=0)
        completion_mask = completion_mask * valid_mask.unsqueeze(1).int()

        # ── Loss (Token-level normalization) ──────────────────────────
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -per_token_loss

        # ── Token-level Loss: 전체 유효 토큰 수로 정규화 ─────────────────────────────────
        total_tokens = completion_mask.sum()
        if total_tokens > 0:
            loss = (per_token_loss * completion_mask).sum() / total_tokens
        else:
            loss = (per_token_loss * completion_mask).sum() * 0.0

        # ── [11] Metrics ───────────────────────────────────────────────
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
