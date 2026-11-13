# NimbusPay ticket triage

Base model: `Qwen/Qwen2.5-0.5B-Instruct`

Pinned revision: `7ae557604adf67be50417f59c2c2f167def9a775`

Final base loaded in 4-bit: **True**

Best run `clean_qlora`: dev exact match **58.5%**, mean field accuracy **92.6%**.

Open notebook.ipynb in a free Colab T4 runtime. Upload the original candidate_data.zip when requested; the training data and base weights are intentionally excluded from Git. Read report.pdf for methods and measured experiments. Source modules are embedded in the notebook and also provided as .py files for inspection.

Data credit: some ticket text is adapted from Banking77 (PolyAI, CC-BY-4.0), as credited in the supplied assignment. The task, categories and labels differ from Banking77. Training data is excluded from this repository.
