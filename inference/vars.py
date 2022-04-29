header = """#!/bin/bash
#SBATCH --job-name=infer-{model_name}-{dataset_name}-{split}
#SBATCH --time=2:00:00
#SBATCH --partition=gpushort
#SBATCH --mem=16GB
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/data/p305238/slurm_logs/%x.%j.out
 
module purge
module load Python/3.8.6-GCCcore-10.2.0
module load CUDA/10.2.89-GCC-8.3.0

cd /data/$ME
source venv/bin/activate
 
python3 --version
which python3

export HF_HOME=/data/$ME/hf_cache/

"""

command = """python3 -u /home/$ME/scripts/inference/infer.py \\
    --model_name_or_path {model} \\
    --use_auth_token <YOUR_AUTH_TOKEN> \\
    --max_source_length {source_len} \\
    --max_target_length {target_len} \\
    --dataset_name it5/datasets \\
    --dataset_config {config} \\
    --dataset_split {split} \\
    --output_dir /scratch/$ME/it5_experiments/preds/ \\
    --batch_size={bs} \\
    --source_column={source_column} \\
    --target_column={target_column}

"""

footer = "deactivate"

settings = {
    "fst_i2f": {
        "source_len": 128,
        "target_len": 128,
        "config": "fst",
        "suffix": "informal-to-formal",
        "source_column": "informal",
        "target_column": "formal",
        "splits": ["test_0"]
    },
    "fst_f2i": {
        "source_len": 128,
        "target_len": 128,
        "config": "fst",
        "suffix": "formal-to-informal",
        "source_column": "formal",
        "target_column": "informal",
        "splits": ["test_0", "test_1", "test_2", "test_3"]
    },
    "hg": {
        "source_len": 512,
        "target_len": 64,
        "suffix": "headline-generation",
        "source_column": "text",
        "target_column": "target"
    },
    "ns": {
        "source_len": 512,
        "target_len": 128,
        "suffix": "news-summarization",
        "source_column": "source",
        "target_column": "target",
        "splits": ["test_fanpage", "test_ilpost"]
    },
    "qa": {
        "source_len": 512,
        "target_len": 64,
        "suffix": "question-answering",
        "source_column": "source",
        "target_column": "target"
    },
    "qg": {
        "source_len": 512,
        "target_len": 128,
        "suffix": "question-generation",
        "source_column": "text",
        "target_column": "target"
    },
    "st_g2r": {
        "source_len": 512,
        "target_len": 64,
        "suffix": "ilgiornale-to-repubblica",
        "source_column": "full_text",
        "target_column": "headline"
    },
    "st_r2g": {
        "source_len": 512,
        "target_len": 64,
        "suffix": "repubblica-to-ilgiornale",
        "source_column": "full_text",
        "target_column": "headline"
    },
    "wits": {
        "source_len": 512,
        "target_len": 256,
        "suffix": "wiki-summarization",
        "source_column": "source",
        "target_column": "summary"
    },
}

models = [
    "it5/mt5-small",
    "it5/mt5-base",
    "it5/it5-small",
    "it5/it5-base",
    "it5/it5-large",
    "it5/it5-efficient-small-el32"
]


def get_bsz(model):
    if model.startswith("it5/it5-small"):
        return 128
    elif model.startswith("it5/it5-base") or "mt5-small" in model or model.startswith("it5/it5-efficient-small-el32"):
        return 64
    elif "mt5-base" in model or model.startswith("it5/it5-large"):
        return 32
    raise Exception(f"Unknown model: {model}")


def get_params(model, config):
    params = settings[config]
    params["bs"] = get_bsz(model)
    return params
