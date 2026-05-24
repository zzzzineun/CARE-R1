# CARE-R1

Official codebase for training and evaluating **CARE-R1**.
---

## Setup

```bash
bash setup.sh
```

This creates a `care-r1` conda environment (Python 3.11), installs the patched `open-r1-multimodal` package.
---

## Training

```bash
bash train_careR1.sh
```

Before running, edit the following fields at the top of `train_careR1.sh`:

| Variable | Description |
|---|---|
| `WANDB_API_KEY` | Your W&B API key |
| `WANDB_ENTITY` | Your W&B entity/team name |
| `HF_TOKEN` | HuggingFace token (for model download) |
| `DATA_DIR` | Root data directory containing `Huatuo/` |
| `OUTPUT_DIR` | Where to write checkpoints and logs |
| `GPU_NUM` | Number of GPUs for `torchrun` |
| `DATASET_SELECTION` | Imaging modality to train on (see below) |

---

## Evaluation

```bash
bash evaluate.sh
```

Before running, edit the top of `evaluate.sh`:

| Variable | Description |
|---|---|
| `HF_TOKEN` | HuggingFace token |
| `MODEL_PATH` | Path to trained model checkpoint |
| `DATA_DIR` | Root data directory containing `OmniMedVQA/` |
| `OUTPUT_DIR` | Where to write results and eval logs |
| `EVAL_SCRIPT` | Path to `src/eval/original_test_qwen2vl_med.py` |

---

## File Structure

```
CARE-R1/
├── setup.sh                        # Environment setup
├── train_careR1.sh                 # Training entry point
├── evaluate.sh                     # Evaluation entry point
│
├── data/
│   └── OmniMedVQA/
│       └── modality/test/          # Per-modality test JSONs
│           ├── CT_test.json
│               ...
│
└── src/
    ├── distill_r1/                 
    ├── eval/
    │   └── original_test_qwen2vl_med.py   # Evaluation script
    │
    └── open-r1-multimodal/         # Patched open-r1 framework
        ├── configs/                # DeepSpeed / accelerate configs
        ├── local_scripts/          # ZeRO JSON configs
        └── src/open_r1/
            ├── care.py             # CARE-R1 training script
            └── trainer/
                ├── grpo_trainer.py
                └── care_trainer.py
```

---

## Acknowledgements

CARE-R1 builds on top of several open-source projects:

- [open-r1-multimodal](https://github.com/EvolvingLMMs-Lab/)
- [R1-V] (https://github.com/StarsfieldAI/R1-V/)
- [MedVLM-R1] (https://github.com/JZPeterPan/MedVLM-R1/)


---

## License

This project is licensed under the Apache 2.0 License.
