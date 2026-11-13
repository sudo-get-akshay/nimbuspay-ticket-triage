# NimbusPay ticket triage

Base model: `Qwen/Qwen2.5-0.5B-Instruct` (494M parameters).

The experiment matrix compares 4-bit QLoRA and 16-bit LoRA. Final loading mode and best score will be populated from measured results after the T4 runs finish.

The training-only audit retains 1,350 of 1,445 rows: 95 drops and 175 repairs. See `cleaning_log.csv` and `audit.py`.

Open `notebook.ipynb` in a free Colab T4 runtime. Upload the original private `candidate_data.zip` when prompted. The notebook includes both baselines, five runs forming four controlled comparisons, dev scoring, final PEFT export, and unmodified model test outputs. Training data and base model weights are excluded from this repository.
