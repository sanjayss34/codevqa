# Modular Visual Question Answering via Code Generation

This repo contains the code for the paper Modular Visual Question Answering via Code Generation, published at ACL 2023.

# Setup
Follow these steps to set up an environment to run this code. First, create a fresh conda environment based on Python 3.8. Then
1. Run `pip install -e .` inside this repo. 
2. Clone the Grounding DINO repo (https://github.com/IDEA-Research/GroundingDINO) and run `python -m pip install -e GroundingDINO` inside it to install it.
3. `pip install transformers==4.25 openai sentence-transformers`
Though the annotations for all 5 datasets used in our paper's evaluations are available online, we collected these annotations (and provided the dataset samples used in our evaluations when applicable) in a single zip file for your convenience: https://drive.google.com/file/d/1FrGEpgcGi9SjLPbQ-bGLlGZrdOAqA79j/view?usp=sharing .

# Experiments
Run these scripts to reproduce the results of CodeVQA and Few-shot PnP-VQA on the GQA, COVR, and NLVR2 test sets.
```
bash run_scripts/pnp-vqa/eval/gqa_eval_gpt3.sh
bash run_scripts/pnp-vqa/eval/covr_eval_gpt3.sh
bash run_scripts/pnp-vqa/eval/nlvr2_eval_gpt3.sh
```
The config files are stored at `lavis/projects/pnp-vqa/eval/{gqa/covr/nlvr2}_eval_gpt3{_codevqa}.yaml`. We provide a few commented-out options for (1) if you want to evaluate on the validation set (or sample thereof) instead of the test set, (2) randomly retrieving in-context examples instead of using question embeddings, and (3) using the `find_object` primitive for counting objects (for this, provided in the COVR and NLVR2 configs, use both the commented-out option for the `programs_path` and the commented-out `grounding_dino_path`).

# Acknowledgements
This repo is based on the original LAVIS repo: https://github.com/salesforce/LAVIS .

# Citation
If you find our paper or this repository useful in your work, please cite our paper:
```
@inproceedings{subramanian-etal-2023-modular,
    title = "Modular Visual Question Answering via Code Generation",
    author = "Subramanian, Sanjay  and
      Narasimhan, Medhini and
      Khangaonkar, Kushal and
      Yang, Kevin and
      Nagrani, Arsha and
      Schmid, Cordelia and
      Zeng, Andy and
      Darrell, Trevor and
      Klein, Dan",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics",
    month = july,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics"
}
```
