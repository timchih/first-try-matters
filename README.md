# First Try Matters: Revisiting the Role of Reflection in Reasoning Models
Implementation for the paper "First Try Matters: Revisiting the Role of Reflection in Reasoning Models"

## Generate rollouts
Use the following command to collect rollouts of a model on 5 datasets: AIME2024, AIME2025, AMC, MATH500, and Olympiad Bench.
```bash
bash generate_rollouts.sh -m PATH_TO_YOUR_MODEL -o rollouts/YOUR_OUTPUT_FOLDER_NAME
```
## Run Candidate Extraction
First specify the rollouts to run extraction in `infer_llm.py` line 407.
Then run the following command.
```bash
bash start_sglang_and_run_gpt.sh --run-id YOUR_RUN_IDENTIFIER --sp-id 3  # choose from 1, 2, 3 for system prompts, see system_prompts.py for details. In the paper we use 3.
```

## Datasets

Datasets curated in Section 3.1 of the paper can be downloaded from 🤗 Hugging Face: https://huggingface.co/datasets/olafyiii/first-try-matters.

## Citation
If you find this work useful, please consider citing:
```bibtex
@article{kang2025first,
  title={First Try Matters: Revisiting the Role of Reflection in Reasoning Models},
  author={Kang, Liwei and Deng, Yue and Xiao, Yao and Mo, Zhanfeng and Lee, Wee Sun and Bing, Lidong},
  journal={arXiv preprint arXiv:2510.08308},
  year={2025}
}
```